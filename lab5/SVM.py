import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import xlrd
import sklearn.metrics as sm

# 进行标准化标签
def Encodeing(X):
    label_encoder = []
    X_encoded = np.empty(X.shape)
    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:, i] = X[:, i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
    return X_encoded,label_encoder

# 数据读取
def read_xls_file(filename):                         #读取训练数据
    X = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line[:-1].split(',')
            data = line[1:]                          #去掉第一列数据
            X.append(data)
    X = np.array(X)

    # 进行标准化标签
    X_encoded,label_encoder= Encodeing(X)

    dataMat = X_encoded[:,:-2].astype(int)
    labels_1 = X_encoded[:,-2].astype(int)
    labels_2 = X_encoded[:,-1].astype(int)
    return dataMat,labels_1,labels_2,label_encoder

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



if __name__ == "__main__":
    dataMat,labels_1,labels_2,label_encoder= read_xls_file('./data/train.csv')

    dataMat_Norm = dataMat
    print(dataMat)
    print(labels_1)
    print(labels_2)

    params = {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.2}
    regressor = SVR(**params)

    regressor.fit(dataMat_Norm,labels_1)

    input_data = ['22.44','0.75','0.11']
    input_data_encoded = [-1] * len(input_data)
    count = 0
    for i, item in enumerate(input_data):
        if item.isdigit():
            input_data_encoded[i] = int(input_data[i])
        else:
            input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
            count = count + 1 

    input_data_encoded = np.array(input_data_encoded)

