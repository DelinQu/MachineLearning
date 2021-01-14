from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

def loadData():                                 # 文件读取函数
    f=open('./data/train.csv')                  # 打开文件    
    data = f.readlines()    
    
    l=len(data)                                 # mat为l*6的矩阵,元素都为0
    mat=zeros((l,6))                            
    index=0                                     
    xdata = ones((l,4))                         #xdata为l*4的矩阵，元素都为1
    ydata1,ydata2= [],[]                        #两列数据结果                      
    for line in data:
        line = line.strip()                     #去除多余字符
        linedata = line.split(',')              #对数据分割
        mat[index, :] = linedata[0:6]           #得到一行数据
        index +=1
    yearData   = mat[:,0]                       # 得到年份                  
    xdata[:,1] = mat[:,1]                       #得到第1列数据
    xdata[:,2] = mat[:,2]                       #得到第2列数据
    xdata[:,3] = mat[:,3]                       #得到第3列数据
    ydata1 = mat[:,4]                           #得到第4列数据
    ydata2 = mat[:,5]                           #得到第5列数据
    print(xdata)
    return yearData,xdata,ydata1,ydata2

# 直线方程
def model(theta,x):                             #直线方程
    theta = np.array(theta)
    return x.dot(theta)

# 代价函数
def cost(theta,xdata,ydata,l):                  #代价函数
    SUM = 0
    idex = 0
    ydata = mat(ydata)
    ydata = ydata.T
    for line in ydata:
          yp = model(theta,xdata[idex ,:])     
          yp = yp - ydata[idex ,:]
          yp = yp**2
          SUM = SUM + yp
          idex  =idex +1
    return SUM/2/l                             #返回代价

# 梯度计算
def grad(theta,idex1,xdata,ydata,sigmal,l):     #梯度计算
    idex = 0
    SUM = 0
    for line in ydata:
            yp = model(theta,xdata[idex ,:]) - ydata[idex]
            yp =yp * xdata[idex][idex1]
            idex  =idex +1
            SUM = SUM + yp
    return SUM/l

# 参数更新
def gradlient(theta,xdata,ydata,sigmal,l):     #参数更新
    index = 0
    for line1 in theta:
            theta[index] = theta[index] - sigmal * grad(theta,index,xdata,ydata,sigmal,l)  #参数更新
            theta[index] = theta[index]
            index = index+1
    return theta                              #返回参数

# 归一化数据
def Min_Max(xdata):                             #归一化数据
    index = 1
    while index < len(xdata[0,:]):
          item = xdata[:,index].max()
          item1 = xdata[:,index].min()
          xdata[:,index] = (xdata[:,index] - item1)/(item-item1)   #归一化数据
          index = index+1
    return xdata

# 梯度下降求解函数
def OLS(xdata,ydata):                           
    theta = [0,0,0,0]                           
    iters = 0
    iters = int(iters)
    l = len(ydata)
    l = int(l)
    cost_record = []
    it = []
    sigmal = 0.1
    cost_val = cost(theta,xdata,ydata,l)                #计算代价
    cost_record.append(cost_val)
    it.append(iters)
    while iters <1500:
          theta = gradlient(theta,xdata,ydata,sigmal,l) #计算梯度
          cost_updata = cost(theta,xdata,ydata,l)       #计算代价
          iters = iters + 1
          cost_val = cost_updata
          cost_record.append(cost_val)                  #记录代价函数
          it.append(iters)
    return mat(theta).T,cost_record,it,theta


def show(pre1,pre2,ydata1,ydata2):                                          # 图形显示
    matplotlib.rcParams['font.sans-serif']=['SimHei']                       # 防止中文乱码                         
    fig,axes = plt.subplots(figsize=(12,10))
    line1, = axes.plot(pre1,'k',markeredgecolor='b',marker = 'o',markersize=7)
    line2, = axes.plot(ydata1,'r',markeredgecolor='g',marker = u'$\star$',markersize=7)
    line3, = axes.plot(pre2,'g',markeredgecolor='g',marker = 'o',markersize=7)
    line4, = axes.plot(ydata2,'y',markeredgecolor='b',marker = u'$\star$',markersize=7)
    
    axes.legend((line1,line2,line3,line4),(u'客运量预测输出',u'客运量真实输出',u'货运量预测输出',u'货运量真实输出'),loc = 'upper left')
    axes.set_ylabel(u'公路客运量及货运量')
    xticks = range(0,22,1)
    xtickslabel = range(1990,2012,1)
    axes.set_xticks(xticks)
    axes.set_xticklabels(xtickslabel)
    axes.set_xlabel(u'年份')
    axes.set_title(u'多元线性回归MLR')

    plt.savefig("./fig/MLR.png")
    plt.show()
    plt.close()

def Normal(test,xcopy):
    test[1] = (test[1]-xcopy[:,1].min())/(xcopy[:,1].max()-xcopy[:,1].min())
    test[2] = (test[2]-xcopy[:,2].min())/(xcopy[:,2].max()-xcopy[:,2].min())
    test[3] = (test[3]-xcopy[:,3].min())/(xcopy[:,3].max()-xcopy[:,3].min())
    return test

def trainPre(xdata,ws1,ws2):                               #预测训练情况
    # 归一化
    xdataNormal=[]
    index = 1
    while index < len(xdata[:,0]):
        item = list(xdata[index,:])
        item=Normal(item,xdata)     
        xdataNormal.append(item)
        index = index+1

    # 预测
    pre1,pre2=[],[]
    for item in xdataNormal:
        pre1.append((item*ws1)[0,0])
        pre2.append((item*ws2)[0,0])

    return pre1,pre2

if __name__ == "__main__":
    yearData,xdata1,ydata1,ydata2=loadData()                     # 加载数据
    xcopy = xdata1.copy()                                

    xdata1 = Min_Max(xdata1)                                     # 归一化数据
    xdata2=xdata1
    
    ws1,cost_rec1,iters1,thet1= OLS(xdata1,ydata1)             # 训练
    ws2,cost_rec2,iters2,thet2= OLS(xdata2,ydata2)
    
    test1 = [1,73.39,3.9635,0.9880]
    test2 = [1,75.55,4.0975,1.0268]

    test1=Normal(test1,xcopy)
    print("2010年预测的公路客运量为：", (test1*ws1)[0,0],"(万人)")
    print("2010年预测的公路货运量为：", (test1*ws2)[0,0],"(万吨)")
  
    test2=Normal(test2,xcopy)
    print("2011年预测的公路客运量为：", (test2*ws1)[0,0],"(万人)")
    print("2011年预测的公路货运量为：", (test2*ws2)[0,0],"(万吨)")
    
    # 测试训练结果
    pre1,pre2=trainPre(xcopy,ws1,ws2)

    show(pre1,pre2,ydata1,ydata2)
    trainPre(xcopy,ws1,ws2)
