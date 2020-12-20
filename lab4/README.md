## 机器学习实验4——DecisionTree 

​																																																				——created by 屈德林 in 2020/12/20

目录：

[TOC]

## 简介

- 本文件是对lab4目录下的DecisionTree代码的介绍，你将会了解lab4目录结构，以及程序运行的方法。



## 运行环境

- 操作系统：`Ubuntu20.04 LST`
- 操作软件：`Pycharm`
- 解释器：`python3.8.2`




## 文件目录

```bash
$ tree
.
├── C45_Decision.py
├── CART_Decision.py
├── data
│   ├── ex3data.csv
│   └── ex3dataEn.csv
├── fig
│   ├── C45.png
│   ├── CART.png
│   ├── ID3.png
│   ├── sklearn
│   ├── sklearn.pdf
│   └── tree.dot
├── ID3_Decision.py
├── README.md
├── Sklearn_Decision.py
└── utils
    ├── plotDecisionTree.py
    └── __pycache__
        └── plotDecisionTree.cpython-38.pyc
```

- `C45_Decision.py`：C4.5求解决策树

- `CART_Decision.py`：CART求解决策树

- `data`：数据集

- `fig`：可视化结果

- `ID3_Decision.py`：ID3算法求解决策树

- `Sklearn_Decision.py`：Sklearn机器学习库求解决策树

- `utils`工具类

  



## 如何运行？

- 使用pythn解释器对程序进行测试

```bash
$ python C45_Decision.py  
$ python CART_Decision.py  
$ python ID3_Decision.py  
$ python Sklearn_Decision.py
```

![image-20201220140536364](https://i.loli.net/2020/12/20/TqeKJbMrgEG9Roc.png)