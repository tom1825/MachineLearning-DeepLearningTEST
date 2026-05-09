import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# 1. 数据加载
def load_imdb_data(data_dir):
    texts = []
    labels = []
    for label in ['pos', 'neg']:
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return texts, labels

print("Loading data...")
train_texts, train_labels = load_imdb_data('aclImdb/train')
test_texts, test_labels = load_imdb_data('aclImdb/test')

# 2. 划分验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# 3. 数据清洗
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\b\w{1,2}\b", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("数据清洗...")
train_texts = [clean_text(t) for t in train_texts]
val_texts   = [clean_text(t) for t in val_texts]
test_texts  = [clean_text(t) for t in test_texts]

# 4. TF-IDF特征
print("Vectorizing...")
#创建向量器
vectorizer = TfidfVectorizer(
    max_features=80000,
    stop_words='english',
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,      #对数变换
    strip_accents='unicode',
)

X_train = vectorizer.fit_transform(train_texts)
X_val   = vectorizer.transform(val_texts)
X_test  = vectorizer.transform(test_texts)

# 5. 模型训练函数
def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    f1  = f1_score(y_val, val_pred)
    print(f"{name} Validation Accuracy: {acc:.4f}")
    print(f"{name} Validation F1-score: {f1:.4f}")
    return model

# 6. 三个模型

# 1️朴素贝叶斯
nb = MultinomialNB(alpha=0.1)
nb = evaluate_model("朴素贝叶斯", nb, X_train, train_labels, X_val, val_labels)

# 2️逻辑回归
lr = LogisticRegression(
    max_iter=1000,
    C=7,
    solver='lbfgs',     #计算最小损失的方法
    penalty='l2',
    n_jobs=-1,
)
lr = evaluate_model("逻辑回归", lr, X_train, train_labels, X_val, val_labels)

# 3️随机森林
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    n_jobs=-1,
    random_state=50,
)
rf = evaluate_model("随机森林", rf, X_train, train_labels, X_val, val_labels)

# 7. 集成模型（Voting）
print("\nTraining Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('nb', nb),
        ('lr', lr),
        #('rf', rf),
    ],
    voting='soft',
    weights=[3, 7],
)
ensemble.fit(X_train, train_labels)

val_pred = ensemble.predict(X_val)
print("\n验证集:")
print("Ensemble Validation Accuracy:", accuracy_score(val_labels, val_pred))
print("Ensemble Validation F1:",       f1_score(val_labels, val_pred))

#test_pred = ensemble.predict(X_test)
#print("测试集:")
#print("Test Accuracy:", accuracy_score(test_labels, test_pred))
#print("Test F1:",        f1_score(test_labels, test_pred))