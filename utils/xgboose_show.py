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

def xgboost_show(model):
    xgboost.plot_importance(model, height=.5,
                            max_num_features=10,
                            show_values=True,
                            importance_type='gain')
    plt.show()

if __name__ == '__main__':
    # 加载数据集
    data = load_iris()
    X = data['data']
    y = data['target']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgboost.XGBClassifier(eval_metric='mlogloss').fit(X, y)

    # 获取特征的重要性值
    importances = model.feature_importances_
    # 获取特征的重要性值
    importances1 = model.get_booster().get_score(importance_type='gain')  # 设置为'gain'
    print(f"importances{importances}")
    print(f"importances1{importances1}")
    xgboost_show(model)