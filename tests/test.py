from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=10, random_state=42)

# 训练随机森林
clf.fit(X_train, y_train)

# 可视化整个随机森林
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=300)
tree.plot_tree(clf.estimators_[0], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()