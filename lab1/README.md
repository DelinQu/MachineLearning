

## 机器学习实验1——线性回归 说明文档 

​																																																				——created by 屈德林 in 2020/11/16

目录：

[TOC]

## 简介

- 本文件是对lab1目录下的一元线性回归和多元线性回归代码的介绍，你将会了解lab1目录结构，以及程序运行的方法。



## 运行环境

- 操作系统：`Ubuntu20.04 LST`
- 操作软件：`Pycharm`
- 解释器：`python3.8.2`

- py库

```python
import numpy as np							# 开源的数值计算扩展。这种工具可用来存储和处理大型矩阵
import pandas as pd							# Pandas 纳入了大量库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具
import matplotlib.pyplot as plt				# Matplotlib 是一个 Python 的 2D绘图库		
```



## 文件目录

```bash
.
├── data
│   ├── ex1data1.txt
│   └── ex1data2.txt
├── figOne
│   ├── fig1.png
│   ├── fig2.png
│   └── fig3.png
├── figTwo
│   ├── fig0.png
│   ├── fig10.png
│   ├── fig11.png
│   ├── fig12.png
│   ├── fig13.png
│   ├── fig14.png
│   ├── fig15.png
│   ├── fig16.png
│   ├── fig17.png
│   ├── fig18.png
│   ├── fig19.png
│   ├── fig1.png
│   ├── fig20.png
│   ├── fig21.png
│   ├── fig2.png
│   ├── fig3.png
│   ├── fig4.png
│   ├── fig5.png
│   ├── fig6.png
│   ├── fig7.png
│   ├── fig8.png
│   └── fig9.png
├── MultipleLinearReg.py
├── README.md
└── UnitarylinearReg.py

3 directories, 30 files
```

- `data`：测试数据集
- `UnitarylinearReg.py`：一元线性回归程序
- `figOne`：一元线性回归程序图像保存目录
- `MultipleLinearReg.py`：多元线性回归程序
- `figTwo`：多元线性回归程序图像保存目录



## 如何运行？

- 键入命令对数据集`ex1data1`进行线性回归拟合和预测：

```bash
$ python ./UnitarylinearReg.py 
```

- 键入命令对数据集`ex1data2`进行线性回归拟合和预测：

```bash
$ python ./MultipleLinearReg.py
```

