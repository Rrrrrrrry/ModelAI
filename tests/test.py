# 导入所需的库
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
class CustomVotingClassifier:
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting
        self.voting_classifier = VotingClassifier(estimators=self.estimators, voting=self.voting)

    def fit(self, X, y):
        self.voting_classifier.fit(X, y)

    def predict(self, X):
        return self.voting_classifier.predict(X)

    def predict_proba(self, X):
        return self.voting_classifier.predict_proba(X)

# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义基本分类器
base_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('knn', KNeighborsClassifier()),
    ('svm', SVC(kernel='linear', probability=True))
]

# 定义次级分类器
meta_classifier = LogisticRegression()

# 定义Stacking分类器
stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier, cv=5)

# 训练Stacking分类器
stacking_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = stacking_classifier.predict(X_test)

# 输出预测结果
print("Stacking Classifier Predictions:")
print(y_pred)

# 输出预测准确率
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
