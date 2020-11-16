import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path="./data/ex1data2.txt"                  # 路径

# 从文件流中获取数据
def readeData(path):                        # 读取数据
    data=pd.read_csv(path,header=None,names=['Size', 'Bedrooms', 'Price'])
    return data

def drawFit(theta,figName):                 # 画出拟合结果
    x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 线性，规定坐标范围
    f = theta[0, 0] + (theta[0, 1] * x)                                 # 计算拟合结果
    fig,ax = plt.subplots(figsize=(12, 8))                              # 确定图大小
    ax.plot(x, f, 'r', label='Prediction')                              # 设置ax的标题
    ax.scatter(data.Population, data.Profit, label='Training Data')     # 散点图
    ax.legend(loc=2)                                                    # 图例
    ax.set_xlabel('Population')                                         # 横坐标标签
    ax.set_ylabel('Profit')                                             # 纵坐标标签
    ax.set_title('Predicted Profit vs. Population Size')                # 设置标题
    fig.savefig("./figTwo/"+figName)                                    # 保存图
    plt.show()                                                          # 作图

def drawCost(cost,figName,alpha):
    plt.figure(figsize=(12, 8))
    plt.scatter(range(1000),cost[0:1000],color='red')   # 绘制迭代过程中的cost代价函数
    plt.xlabel("iterations")                # x标题
    plt.ylabel("costFunction")              # y标题
    plt.title("alpha = "+alpha+"Iterations and Costfunction line")   # 图标题
    plt.savefig("./figTwo/"+figName)        #保存
    plt.show()

# 计算代价函数J(θ)
def getCostFunc(x,y,theta):
    inner=np.power(((x*theta.T)-y),2)
    return np.sum(inner) / (2*len(x))

def Init(data):
    data.insert(0, "Theta0", 1)             # 新增一列，保存Theta0的值
    cols = data.shape[1]                    # 列数
    x = data.iloc[:, 0:cols - 1]            # x是去掉最后一列的矩阵       97*2
    y = data.iloc[:,cols-1:cols]            # y是最后一列的矩阵          97*1

    x = np.mat(x.values)                    # 转化成np矩阵类型
    y = np.mat(y.values)
    theta = np.mat(np.array([0, 0,0]))      # theta为[[0,0]],初始化为0  1*3,保证可以和x做运算，第一个矩阵的列的个数等于第二个矩阵的行的个数
    return x,y,theta

# 梯度下降
def GradientDes(x, y, theta, alpha, iters):
    cur = np.mat(np.zeros(theta.shape))     # 用0填充的数组
    cost = np.zeros(iters)                  # 预定义iter个损失值，初始化为0

    #每次迭代计算一次损失值，并赋值
    for i in range(iters):                  # 迭代iters次
        e = (x * theta.T) - y               # 误差矩阵e
        for j in range(theta.shape[1]):     # 更新theta的每一列
            term = np.multiply(e, x[:,j])   # 将误差e与x矩阵的第j列相乘
            cur[0,j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))   # 更新cur[0,j]也就是更新Theta0，迭代公式见文档
        theta = cur                         # 取出cur
        cost[i] = getCostFunc(x, y, theta)  # 计算损失值并保存
    return theta, cost

# 预测分析
def getPredict(p,theta):
    return p*theta.T

# 归一化
def Normalization(x,data):
    x = (x- data.mean()) / data.std()
    return x

def Normalization2(x,data):
    x = (x- data.min()) / (data.max()-data.min)
    return x

def findAlpha(x, y, theta):
    alphaList=[0.0001,0.0002,0.0004,0.0006,0.0008,      # alpha的序列值
               0.001,0.002,0.004,0.006,0.008,
               0.01,0.02,0.04,0.06,0.08,
               0.1,0.2,0.4,0.6,0.8,
               1,1.5]
    iters=1500
    i=0
    for alpha in alphaList[:-1]:
        curTheta, cost = GradientDes(x, y, theta, alpha, iters)
        drawCost(cost,"fig"+str(i)+".png",str(alpha))
        i=i+1
    # 1.5的时候已经不收敛
    curTheta, cost = GradientDes(x, y, theta, 1.5, 1000)
    drawCost(cost, "fig" + str(i) + ".png", str(1.5))

if __name__ == '__main__':
    # 数据读入
    data=readeData(path)

    # 归一化
    NOrData=data
    NorData=Normalization(NOrData,data)

    # 初始化
    x,y,theta=Init(NorData)

    findAlpha(x,y,theta)

    # 梯度下降
    alpha=0.02
    iters=1500
    theta,cost=GradientDes(x, y, theta, alpha, iters)
    print("迭代过程 \ncost=", cost)
    print("\n迭代结果\nTheta=",theta)


    # 预测分析
    p=[1,1650,3]
    p[1] = (p[1] - data.values.mean()) / data.values.std()
    p[2] = (p[2] - data.values.mean()) / data.values.std()
    print(p)
    pre1=getPredict(p,theta)*data.values.std()+data.values.mean()
    print("\n预测结果=",pre1)

