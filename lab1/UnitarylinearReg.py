import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path="./data/ex1data1.txt"                  # 路径

# 从文件流中获取数据
def readeData(path):                        # 读取数据
    data=pd.read_csv(path,header=None,names=["Population","Profit"])
    return data

# 作图
def drawScatter(data,figName):              # 画出散点图
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
    plt.title("Scatter chart of profit population")
    plt.savefig("./figOne/"+figName)           #保存
    plt.show()

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
    fig.savefig("./figOne/"+figName)                                       # 保存图
    plt.show()                                                          # 作图

def drawCost(cost,figName):
    plt.figure(figsize=(12, 8))
    plt.plot(range(1000),cost[0:1000])      # 绘制迭代过程中的cost代价函数
    plt.xlabel("iterations")                # x标题
    plt.ylabel("costFunction")              # y标题
    plt.title("Iterations and Costfunction line")   # 图标题
    plt.savefig("./figOne/"+figName)           #保存
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
    theta = np.mat(np.array([0, 0]))        # theta为[[0,0]],初始化为0  1*2,保证可以和x做运算，第一个矩阵的列的个数等于第二个矩阵的行的个数
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

if __name__ == '__main__':
    # 数据读入
    data=readeData(path)

    # 初始化
    x,y,theta=Init(data)

    # 计算代价函数
    cost=getCostFunc(x,y,theta)
    print("初始代价函数值",cost)

    # 设定学习速率α和要执行的迭代次数iters
    alpha, iters=0.01,1500

    # 梯度下降
    theta,cost=GradientDes(x, y, theta, alpha, iters)
    print("\n迭代结果Theta=",theta)
    print("迭代过程 cost=",cost)

    # 绘制图像
    drawScatter(data, "fig1.png")
    drawCost(cost, "fig2.png")
    drawFit(theta,"fig3.png")

    # 预测分析
    pre1=getPredict([1,1.35],theta)
    pre2=getPredict([1,7],theta)
    print("\n预测结果1=",pre1)
    print("预测结果2=",pre2)
