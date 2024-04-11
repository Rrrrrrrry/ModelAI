import os
import sys

from sklearn.datasets import load_iris

currentdir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(currentdir)
from utils.model_utils import *
from algorithms.machine_learning.ensemble.decisionTree_model import *
if __name__=='__main__':
    """
    数据获取
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # X_train = [[0, 0], [1, 1], [2, 2], [3, 3]]
    # y_train = [0, 1, 2, 3]
    # X_test = [[4, 4], [5, 5]]
    # feature_names = ['one', 'two']
    # class_names = ['0', '1', '2', '3']

    # 训练模型
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'splitter': ['best', 'random']
    }

    model_save_path = os.path.join(sys.path[0], 'model')
    os.makedirs(model_save_path, exist_ok=True)

    DT = DecisionTree()
    DT.train(X_train, y_train)
    # DT.train(datatrain, labelstrain, criterion='gini', max_depth=3, min_samples_split=3)
    # DT.grid_search(datatrain, labelstrain, param_grid, cv=10)
    print(f"DT:{DT.model.get_params()}")
    print(f"DT.model.classes_:{DT.model.classes_}")
    # 保存模型
    os.makedirs(model_save_path, exist_ok=True)
    save_model(DT, os.path.join(model_save_path, "DecisionTree_model.joblib"))
    # 读取模型
    DT = load_model(os.path.join(model_save_path, "DecisionTree_model.joblib"), data_format='joblib')
    pre_data, acc = DT.predict(X_test, y_test)
    print(pre_data)
    print(f"acc:{acc}")
    """
    结果可视化
    """
    # DT.visualize_tree()
    DT.save_tree_to_pdf(os.path.join(model_save_path, "DecisionTree_model_show.pdf"),
                        feature_names=feature_names, class_names=class_names)

