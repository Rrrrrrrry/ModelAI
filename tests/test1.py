import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost

import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt

# 加载数据集
data = load_iris()
X = data['data']
y = data['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# 使用LIME解释模型
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=data.feature_names,
    class_names=data.target_names,
    mode='classification'
)

# 解释一个样本
idx = 5
exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=5)
print('LIME Explanation:')
print(exp.as_list())
exp.show_in_notebook(show_table=True)
plt.show()

# 使用SHAP解释模型
shap_explainer = shap.Explainer(model, X_train)
shap_values = shap_explainer(X_test, check_additivity=False)
# 显示SHAP的beeswarm图
shap.summary_plot(shap_values.values, X_test, plot_type="bar")
plt.title('SHAP Feature Importance')
plt.show()

# 显示特定样本的SHAP值
shap.initjs()
shap.force_plot(shap_explainer.expected_value[0], shap_values[idx].values, X_test[idx])