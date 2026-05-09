#California Housing Machine Learning Project

本项目基于 sklearn 自带的 California Housing 数据集，使用多种机器学习方法对加州房屋数据进行分析，包括房价预测、房屋分类和聚类分析。

##Dataset
数据来源于 sklearn 内置的 California Housing 数据集，共包含 20640 条样本和 8 个房屋相关特征，例如：
* 居民收入（MedInc）
* 房屋年龄（HouseAge）
* 平均房间数（AveRooms）
* 人口数量（Population）
* 经纬度（Latitude / Longitude）

预测目标为：
* MedHouseVal（房屋价格中位数）

##Tasks
本项目完成了以下三个任务：

###1. 房价预测
根据房屋特征预测房价。

使用模型：
* Linear Regression（线性回归）
* Decision Tree Regressor（回归树）
评价指标：
* MAE
* RMSE
* R²

###2. 高价房分类（Classification）
将房价转换为二分类标签：

* 高于训练集房价中位数：高价房（1）
* 否则：普通房（0）
使用模型：
* KNN
* SVM
* Decision Tree

评价指标：
* Accuracy
* Precision
* Recall
* F1-score

###3. 房源聚类分析（Clustering）
对房屋数据进行无监督聚类分析。

使用模型：
* KMeans
聚类评估方法：
* Elbow Method
* Silhouette Score

##Project Goal
通过回归、分类和聚类三种机器学习任务，学习并实践监督学习与无监督学习在真实房屋数据中的应用。
