from numpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as ai
import numpy as np
import time

trainPath="./iris/iris_train.data"
testPath="./iris/iris_test.data"


# 加载数据集
def loadData(path):
    dataMat = []
    labelMat1 = []
    labelMat2 = []
    labelMat3 = []
    ylabel = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
        if (lineArr[4] == 'Iris-setosa'):
            labelMat1.append(float(1))
        else:
            labelMat1.append(float(-1))
        if (lineArr[4] == 'Iris-versicolor'):
            labelMat2.append(float(1))
        else:
            labelMat2.append(float(-1))
        if (lineArr[4] == 'Iris-virginica'):
            labelMat3.append(float(1))
        else:
            labelMat3.append(float(-1))
        ylabel.append(lineArr[4])
    return dataMat, labelMat1, labelMat2, labelMat3, ylabel


# 在m中随机选择除了i之外剩余的数
def selectJrand(i, m):
    j = i  # 排除i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

# 修建alpha的值到L和H之间.
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 简化版SMO算法实现
def smoSimple(dataMatrix, classLabels, C, toler, maxIter):
    labelMat = mat(classLabels).T
    b = -1;
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0                   # alpha是否已经进行了优化
        for i in range(m):
            #   w = alpha * y * x;
            #   "SVM分类器函数 y = w^T*x + b"
            #   预测的类别
            fXi = float(multiply(alphas, labelMat).T * dataMatrix * dataMatrix[i, :].T) + b
            Ei = fXi - float(labelMat[i])       # 计算误差，如果误差太大，检查是否可能被优化
            # 满足约束
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 对原始解进行修剪
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:  # print "L==H";
                    continue
                # Eta = -(2 * K12 - K11 - K22)，且Eta非负，此处eta = -Eta则非正
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j,:] * dataMatrix[j, :].T
                if eta >= 0:
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 如果内层循环通过以上方法选择的α_2不能使目标函数有足够的下降，那么放弃α_1
                if (abs(alphas[j] - alphaJold) < 0.00001):  # print "j not moving enough";
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])


                # 更新阈值b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
    return b, alphas

# 根据获取的 [alpha] ，数据点以及标签来获取 [w] 的值:
def calcWs(alphas, dataMatrix, labelMat):
    m, n = shape(dataMatrix)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], dataMatrix[i, :].T)
    return w

# 正确率检测
def pred(dataMat, labelMat, w1, b1, w3, b3):
    dataMat = mat(dataMat)
    sum1 = 0
    m, n = shape(dataMat)
    for i in range(m):
        if (dataMat[i] * w1 + b1 > 0.0 and labelMat[i] == 'Iris-setosa'):
            sum1 += 1
        elif (dataMat[i] * w3 + b3 > 0.0 and labelMat[i] == 'Iris-virginica'):
            sum1 += 1
        elif (dataMat[i] * w3 + b3 < 0.0 and dataMat[i] * w1 + b1 < 0.0 and labelMat[i] == 'Iris-versicolor'):
            sum1 += 1
    m = float(sum1) / float(m) * 100
    print("正确率为： ", m)


if __name__ == '__main__':
    # 加载训练数据集
    xdata, ydata1, ydata2, ydata3, ylabe = loadData(trainPath)
    # 加载测试数据集
    xdata_test,t1,t2,t3,ylabe_test = loadData(testPath)
    # 转换成矩阵
    xdata = mat(xdata)
    xdata_test = mat(xdata_test)

    # 计算
    b1, alphas1 = smoSimple(xdata, ydata1, 0.8, 0.0001, 40)
    # b2 , alphas2 = smoSimple(X,ydata2,0.8,0.0001,40)
    b3, alphas3 = smoSimple(xdata, ydata3, 0.8, 0.0001, 40)
    # 计算
    w1 = calcWs(alphas1, xdata, ydata1)
    # w2 = calcWs(alphas2,X,ydata2)
    w3 = calcWs(alphas3, xdata, ydata3)
    # 检测
    pred(xdata_test, ylabe_test, w1, b1, w3, b3)
