import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,silhouette_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   


# 1. 数据加载 
data = fetch_california_housing()
X = data.data.copy()
y = data.target
# 对数变换
X[:,0] = np.log1p(X[:,0])  # 中位收入
X[:,4] = np.log1p(X[:,4])  # 人口数量
X[:,5] = np.log1p(X[:,5])  # 平均居住人数
# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




# 2. 回归任务
def eval_reg(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

print("===== 回归任务 =====")
# 线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
eval_reg("线性回归", y_test, y_pred_lr)
# 回归树
tree = DecisionTreeRegressor(max_depth=10, min_samples_split=20,min_samples_leaf=10,random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
eval_reg("回归树", y_test, y_pred_tree)




# 3. 分类任务
print("\n===== 分类任务 =====")
#标签分类
median_price = np.median(y_train)
y_train_cls = (y_train > median_price).astype(int)
y_test_cls = (y_test > median_price).astype(int)
# 标准化
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 模型
knn = KNeighborsClassifier(n_neighbors=15)
svm = SVC(C=100, gamma='scale')
dt = DecisionTreeClassifier(max_depth=8, random_state=42)
models = {
    "KNN": knn,
    "SVM": svm,
    "DecisionTree": dt
}

for name, model in models.items():
    model.fit(X_train_std, y_train_cls)
    y_pred = model.predict(X_test_std)
    print(f"\n{name}结果：")
    print(classification_report(y_test_cls, y_pred))

# 混淆矩阵（SVM）
y_pred_svm = svm.predict(X_test_std)
cm = confusion_matrix(y_test_cls, y_pred_svm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("混淆矩阵 (SVM)")
plt.show()




# 4. 聚类任务
print("\n===== 聚类任务 =====")
# 标准化
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X)

K_range = range(1, 10)
inertias = []
silhouettes = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    #(肘部法)
    inertias.append(kmeans.inertia_)
    #(轮廓系数法)
    if k >= 2:
        score=silhouette_score(X_scaled,labels)
        silhouettes.append(score)
        print(f"K={k},inertia={kmeans.inertia_:.4f},silhouette={score:.4f}")
    else:
        silhouettes.append(None)
        print(f"K={k},inertia={kmeans.inertia_:.4f}")
    
# 肘部法
plt.figure()
plt.plot(list(K_range), inertias, marker='o')
plt.xlabel("K")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method")
plt.grid()
plt.show()
# 轮廓系数
plt.figure()
plt.plot(list(K_range)[1:], silhouettes[1:], marker='o')
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.grid()
plt.show()

best_k = list(K_range)[1:][np.argmax(silhouettes[1:])]
print("\n推荐最佳K =", best_k)
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
# 输出每类数量
unique, counts = np.unique(labels, return_counts=True)
for i in range(len(unique)):
    print(f"第{i}类样本数量: {counts[i]}")

