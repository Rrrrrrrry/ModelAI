from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# 加载示例数据集
iris = load_iris()
X = iris.data
y = iris.target

# 创建随机森林分类器模型
rf = RandomForestClassifier().fit(X, y)

# 计算特征的排列重要性
perm_importance = permutation_importance(rf, X, y)

# 获取特征排列重要性得分
importances = perm_importance.importances_mean

# 输出特征排列重要性得分
for i, feature in enumerate(iris.feature_names):
    print(f"Feature {feature}: Importance {importances[i]}")

# 绘制特征排列重要性条形图
plt.bar(iris.feature_names, importances)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Permutation Importance')
plt.show()