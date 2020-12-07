## 机器学习实验3——SVM 

​																																																				——created by 屈德林 in 2020/12/07

目录：

[TOC]

## 简介

- 本文件是对lab3目录下的SVM代码的介绍，你将会了解lab3目录结构，以及程序运行的方法。



## 运行环境

- 操作系统：`Ubuntu20.04 LST`
- 操作软件：`Pycharm`
- 解释器：`python3.8.2`




## 文件目录

```bash
$ tree
.
├── iris
│   ├── iris.data
│   ├── iris_test.data
│   └── iris_train.data
├── READMD.md
├── SVM_PCA.py
└── SVM.py

1 directory, 6 files
```

- `iris`：测试数据集

- `SVM.py`: SMO算法求解SVM

- `SVM_PCA.py`: PCA降维求解并做图

  



## 如何运行？

- 键入命令对数据集iris进行训练和测试

```bash
$ python SVM_PCA.py 
$ python SVM.py 
```

