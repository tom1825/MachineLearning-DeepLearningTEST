# IMDb Movie Review Sentiment Classification

## 项目简介
本项目基于 IMDb 大型电影评论数据集，使用多种传统机器学习方法完成文本情感二分类任务（正面 / 负面）。
通过文本清洗、TF-IDF特征提取，并分别训练朴素贝叶斯、逻辑回归、随机森林模型，同时构建集成模型进行性能提升与对比分析。

## 数据集
使用数据集：IMDb Large Movie Review Dataset
```
aclImdb/
├── train/
│   ├── pos/   # 正面评论
│   ├── neg/   # 负面评论
├── test/
│   ├── pos/
│   ├── neg/
```
* 训练集：25,000 条
* 测试集：25,000 条
* 总计：50,000 条电影评论

## 任务目标
对电影评论进行情感分类：
* 1 → Positive（正面）
* 0 → Negative（负面）
并在验证集上达到：
* Accuracy ≥ 0.90
* F1-score ≥ 0.90

## 处理流程
### 1️数据预处理
* 去除 HTML 标签
* 小写化
* 去除非字母字符

### 2️特征工程
* TF-IDF 向量化
* 去除高频词与低频词

### 3️模型构建
使用三种模型进行对比：
* 朴素贝叶斯（Naive Bayes）
* 逻辑回归（Logistic Regression）
* 随机森林（Random Forest）

### 4️集成学习
使用 VotingClassifier：
* Soft Voting（概率平均）
* 提升整体稳定性

### 5模型评估指标

在验证集上使用：
* Accuracy
* F1-score

## 结果与分析
```
朴素贝叶斯 Validation Accuracy: 0.8792
朴素贝叶斯 Validation F1-score: 0.8798

逻辑回归 Validation Accuracy: 0.9014
逻辑回归 Validation F1-score: 0.9030

随机森林 Validation Accuracy: 0.8638
随机森林 Validation F1-score: 0.8640

Ensemble Validation Accuracy: 0.9026
Ensemble Validation F1: 0.9040772109513492
```

* 逻辑回归在 TF-IDF 特征下表现最佳
* 朴素贝叶斯受“特征独立性假设”限制
* 随机森林不适合高维稀疏文本数据
* 集成学习可提升整体稳定性
