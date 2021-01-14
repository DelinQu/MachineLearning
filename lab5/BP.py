from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import xlrd

# 数据读取
def read_xls_file(filename):                         #读取训练数据  
    data = xlrd.open_workbook(filename)                
    sheet1 = data.sheet_by_index(0)                    
    m = sheet1.nrows                                    
    n = sheet1.ncols                      
    # 人口数量 机动车数量 公路面积 公路客运量 公路货运量              
    pop,veh,roa,pas,fre=[],[],[],[],[] 
    for i in range(m):                                  
        row_data = sheet1.row_values(i)               
        if i > 0:
           pop.append(row_data[1])
           veh.append(row_data[2])
           roa.append(row_data[3])
           pas.append(row_data[4])
           fre.append(row_data[5])
    dataMat = np.mat([pop,veh,roa])
    labels = np.mat([pas,fre])
    dataMat_old = dataMat
    labels_old = labels
    # 数据集合，标签集合，保留数据集合，保留标签集合
    return dataMat,labels,dataMat_old,labels_old

# 读取测试集
def read_xls_testfile(filename):                           #读取测试数据
    data = xlrd.open_workbook(filename) 
    sheet1 = data.sheet_by_index(0)            
    m = sheet1.nrows                           
    n = sheet1.ncols                                    
    pop = []                         
    veh = []
    roa = []        
    for i in range(m):                       
        row_data = sheet1.row_values(i)       
        if i > 0:
           pop.append(row_data[1])
           veh.append(row_data[2])
           roa.append(row_data[3])

    dataMat = np.mat([pop,veh,roa])
    return dataMat

# 由于数据数量级相差太大，我们要先对数据进行归一化处理Min-Max归一化
def Norm(dataMat,labels):
    dataMat_minmax = np.array([dataMat.min(axis=1).T.tolist()[0],dataMat.max(axis=1).T.tolist()[0]]).transpose() 
    dataMat_Norm = ((np.array(dataMat.T)-dataMat_minmax.transpose()[0])/(dataMat_minmax.transpose()[1]-dataMat_minmax.transpose()[0])).transpose()
    labels_minmax  = np.array([labels.min(axis=1).T.tolist()[0],labels.max(axis=1).T.tolist()[0]]).transpose()
    labels_Norm = ((np.array(labels.T).astype(float)-labels_minmax.transpose()[0])/(labels_minmax.transpose()[1]-labels_minmax.transpose()[0])).transpose()
    return dataMat_Norm,labels_Norm,dataMat_minmax,labels_minmax

# 激活函数
def sigmod(x):
    return 1/(1+np.exp(-x))

# Back Propagation
def BP(sampleinnorm, sampleoutnorm,hiddenunitnum=3):                       
    # 超参数
    maxepochs = 60000                                       # 最大迭代次数
    learnrate = 0.030                                       # 学习率
    errorfinal = 0.65*10**(-3)                              # 最终迭代误差
    indim = 3                                               # 输入特征维度3
    outdim = 2                                              # 输出特征唯独2
    # 隐藏层默认为3个节点，1层
    n,m = shape(sampleinnorm)
    w1 = 0.5*np.random.rand(hiddenunitnum,indim)-0.1        #8*3维
    b1 = 0.5*np.random.rand(hiddenunitnum,1)-0.1            #8*1维
    w2 = 0.5*np.random.rand(outdim,hiddenunitnum)-0.1       #2*8维
    b2 = 0.5*np.random.rand(outdim,1)-0.1                   #2*1维

    errhistory = []

    for i in range(maxepochs):
        # 激活隐藏输出层
        hiddenout = sigmod((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose()
        # 计算输出层输出
        networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()
        # 计算误差
        err = sampleoutnorm - networkout
        # 计算代价函数（cost function）sum对数组里面的所有数据求和，变为一个实数
        sse = sum(sum(err**2))/m                                
        errhistory.append(sse)
        if sse < errorfinal:                                    #迭代误差
          break
        # 计算delta
        delta2 = err
        delta1 = np.dot(w2.transpose(),delta2)*hiddenout*(1-hiddenout)
        # 计算偏置
        dw2 = np.dot(delta2,hiddenout.transpose())
        db2 = 1 / 20 * np.sum(delta2, axis=1, keepdims=True)

        dw1 = np.dot(delta1,sampleinnorm.transpose())
        db1 = 1/20*np.sum(delta1,axis=1,keepdims=True)

        # 更新权值
        w2 += learnrate*dw2
        b2 += learnrate*db2
        w1 += learnrate*dw1
        b1 += learnrate*db1

    return errhistory,b1,b2,w1,w2,maxepochs


def show(sampleinnorm,sampleoutminmax,sampleout,errhistory,maxepochs):      # 图形显示
    matplotlib.rcParams['font.sans-serif']=['SimHei']                       # 防止中文乱码                         
    hiddenout = sigmod((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose()
    networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()
    diff = sampleoutminmax[:,1]-sampleoutminmax[:,0]
    networkout2 = networkout
    networkout2[0] = networkout2[0]*diff[0]+sampleoutminmax[0][0]
    networkout2[1] = networkout2[1]*diff[1]+sampleoutminmax[1][0]
    sampleout = np.array(sampleout)

    fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(12,10))
    line1, = axes[0].plot(networkout2[0],'k',markeredgecolor='b',marker = 'o',markersize=7)
    line2, = axes[0].plot(sampleout[0],'r',markeredgecolor='g',marker = u'$\star$',markersize=7)
    line3, = axes[0].plot(networkout2[1],'g',markeredgecolor='g',marker = 'o',markersize=7)
    line4, = axes[0].plot(sampleout[1],'y',markeredgecolor='b',marker = u'$\star$',markersize=7)
    axes[0].legend((line1,line2,line3,line4),(u'客运量预测输出',u'客运量真实输出',u'货运量预测输出',u'货运量真实输出'),loc = 'upper left')
    axes[0].set_ylabel(u'公路客运量及货运量')
    xticks = range(0,22,1)
    xtickslabel = range(1990,2012,1)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xtickslabel)
    axes[0].set_xlabel(u'年份')
    axes[0].set_title(u'BP神经网络')

    errhistory10 = np.log10(errhistory)
    minerr = min(errhistory10)
    plt.plot(errhistory10)
    axes[1]=plt.gca()
    axes[1].set_yticks([-2,-1,0,1,2,minerr])
    axes[1].set_yticklabels([u'$10^{-2}$',u'$10^{-1}$',u'$1$',u'$10^{1}$',u'$10^{2}$',str(('%.4f'%np.power(10,minerr)))])
    axes[1].set_xlabel(u'训练次数')
    axes[1].set_ylabel(u'误差')
    axes[1].set_title(u'误差曲线')
    plt.savefig("./fig/BP6.png")
    plt.show()

    plt.close()
    
    return diff, sampleoutminmax

def pre(dataMat,dataMat_minmax,diff,sampleoutminmax,w1,b1,w2,b2):          #数值预测
    # 归一化数据
    dataMat_test = ((np.array(dataMat.T)-dataMat_minmax.transpose()[0])/(dataMat_minmax.transpose()[1]-dataMat_minmax.transpose()[0])).transpose() 
    # 然后计算两层的输出结果
    # 隐藏层
    hiddenout = sigmod((np.dot(w1,dataMat_test).transpose()+b1.transpose())).transpose()
    # 输出层
    networkout1 = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()
    networkout = networkout1
    # 计算结果
    networkout[0] = networkout[0]*diff[0] + sampleoutminmax[0][0]
    networkout[1] = networkout[1]*diff[1] + sampleoutminmax[1][0]

    print("2010年预测的公路客运量为：", int(networkout[0][0]),"(万人)")
    print("2010年预测的公路货运量为：", int(networkout[1][0]),"(万吨)")
    print("2011年预测的公路客运量为：", int(networkout[0][1]),"(万人)")
    print("2011年预测的公路货运量为：", int(networkout[1][1]),"(万吨)")

if __name__ == "__main__":
    dataMat,labels,dataMat_old,labels_old = read_xls_file('./data/1.xls')
    dataMat_Norm,labels_Norm, dataMat_minmax, labels_minmax = Norm(dataMat,labels)
    err, b1, b2, w1, w2,maxepochs = BP(dataMat_Norm,labels_Norm,6)


    dataMat_test = read_xls_testfile('./data/2.xls')
    diff, sampleoutminmax = show(dataMat_Norm,labels_minmax,labels,err,maxepochs)
    pre(dataMat_test,dataMat_minmax,diff, sampleoutminmax ,w1,b1,w2,b2)