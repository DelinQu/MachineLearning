import numpy as np
import matplotlib.pyplot as plt
path='D:\\课程\\专业课程\\机器学习\\实验二\\logic'

def readData():
    X = []
    Y = []
    with open('logistic_regression_data.txt', 'r') as file:
        for line in file.readlines():
            cur_line = line.strip().split(',')
            X.append([1.0,  float(cur_line[0]), float(cur_line[1])])
            Y.append(float(cur_line[-1]))
    X = np.asarray(X, dtype=np.double)
    N = X.shape[0]
    Y = np.asarray(Y, dtype=np.double)

    X1_0 = []
    X2_0 = []
    X1_1 = []
    X2_1 = []
    for i, t in enumerate(Y):
        if (t == 0):

            X1_0 += [X[:, 1][i]]
            X2_0 += [X[:, 2][i]]
        else:
            X1_1 += [X[:, 1][i]]
            X2_1 += [X[:, 2][i]]

    plt.scatter(X1_0, X2_0, s=30, c='red', marker='s')
    plt.scatter(X1_1, X2_1, s=30, c='green')
    plt.show()
    return N, X, Y

#sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#构造代价函数

def Loss(W):
    loss = 0.0
    for i in range(N):
        loss += -Y[i] * np.log(sigmoid(np.inner(W, X[i]))) - (1 - Y[i]) * np.log(1 - sigmoid(np.inner(W, X[i])))
    return loss


def GD(X, Y, eta, tol):
    W = np.array([0.0, 0.0, 0.0])
    error = 10
    steps = 0
    stepL = []
    errorL = []
    max_cycle = 10000
    while ((error > tol) & (steps < max_cycle)):
        l1 = Loss(W)
        steps += 1
        deltaW = 0
        deltaW += gradL(X, Y, W)
        W = W - eta * deltaW
        l2 = Loss(W)
        error = np.abs(l2 - l1)
        stepL.append(steps)
        errorL.append(error)
    plt.scatter(stepL, errorL)
    plt.savefig(path)
    plt.show()
    return W, error, steps


def gradL(X, Y, W):
    Y = np.squeeze(Y)
    D = sigmoid(np.inner(W, X)) - Y
    G = np.inner(X.T, D)
    return G


def hessian(X, N, W):
    h_w = []
    for i in range(N):
        h_w += [sigmoid(np.inner(W, X[i])) * (1 - sigmoid(np.inner(W, X[i])))]

    h_w = np.asarray(h_w)
    B = np.diag(h_w)
    H = np.dot(X.T, np.dot(B, X))
    return H

#牛顿法求解
def Newton(X, Y, tol):   #tol迭代的最大次数
    W = np.array([0.0, 0.0, 0.0])
    error = 10
    steps = 0
    stepL = []
    errorL = []
    while (error > tol):
        l1 = Loss(W)
        steps += 1
        for i in range(N):
            H = hessian(X, N, W)
            G = gradL(X, Y, W)
            W = W - np.dot(np.linalg.pinv(H), G)
        l2 = Loss(W)
        error = np.abs(l2 - l1)
        stepL.append(steps)
        errorL.append(error)
    plt.scatter(stepL, errorL)
    plt.savefig(path)
    plt.show()
    return W, error, steps

#cost函数 @矩阵叉乘
def cost(theta, X, y):
    costf = np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))
    return costf

if __name__ == '__main__':

    N, X, Y = readData()

    theta = np.zeros(3)
    print('cost = ', cost(theta, X, Y))

    W, error, steps = GD(X, Y, 0.00001, 10 ** (-4))
    print("\nGradient Descent\n", W, "\nError: ", error, " Steps: ", steps)

    W2, error, steps = Newton(X, Y, 10 ** (-5))
    print("\nNewton's\n", W2, "\nError: ", error, " Steps: ", steps)

    X1_0 = []
    X2_0 = []
    X1_1 = []
    X2_1 = []
    for i, t in enumerate(Y):
        if (t == 0):

            X1_0 += [X[:, 1][i]]
            X2_0 += [X[:, 2][i]]
        else:
            X1_1 += [X[:, 1][i]]
            X2_1 += [X[:, 2][i]]

    plt.scatter(X1_0, X2_0, s=30, c='red', marker='s')
    plt.scatter(X1_1, X2_1, s=30, c='green')
    X1 = X[:, 1]

    T1 = - W2[1] / W2[2] * X1
    T2 = [-W2[0] / W2[2] for i in range(X.shape[0])]
    Y2 = T2 + T1
    plt.xlabel('1st Exam Score')
    plt.ylabel('2nd Enam Score')
    plt.plot(X1, Y2, '-.')
    plt.savefig(path)
    plt.show()

    ## new point 1
    X = np.array([1.0,85, 42])
    print('预测的概率为：',sigmoid(np.inner(W2, X)))


