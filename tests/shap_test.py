import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# 加载数据集
data = datasets.load_iris()
X = data['data']
y = data['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
# 使用 SHAP 解释模型
shap_explainer = shap.Explainer(model, X_train)
shap_values = shap_explainer(X_test, check_additivity=False)
shap_values_mean = np.mean(shap_values.values, axis=2)
shap_values_mean_explanation = shap.Explanation(values=shap_values_mean,
                                                base_values=shap_values.base_values[:, 0],
                                                data=X_test)
# 打印形状和类型
print("Shape of shap_values:", shap_values.shape)
print("Type of shap_values:", type(shap_values))
print("Shape of X_test:", X_test.shape)

# 绘制条形图
shap.plots.bar(shap_values_mean_explanation)
plt.show()