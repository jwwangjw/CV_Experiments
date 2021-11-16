# cv作业四-人脸识别

[toc]

## 使用网络

facenet

## 运行环境

|   使用工具包   |   版本   |
| :------------: | :------: |
|    sklearn     | 最新版本 |
| tensorflow-gpu |  1.13.1  |
|     keras      |  2.1.6   |

## 使用说明

### 1.将自己训练集放入datasets文件夹中

注意，datasets文件夹下一级为多个不同类别文件夹，文件夹名为对应类别名

### 2. 使用txt_annotation.py进行文件生成

文件为所有文件类别以及路径，用于训练集与测试集划分

### 3.下载lfw文件，创建lfw文件夹

如训练过程中不想使用lfw进行验证，则在train中进行跳过

### 4.运行train.py

进行训练

### 5.运行predict.py

进行验证以及生成结果文件

## Reference

1.**https://blog.csdn.net/gaobing1993/article/details/108328292**

2.**https://blog.csdn.net/lq126126/article/details/80776105**

3.**https://github.com/davidsandberg/facenet**

