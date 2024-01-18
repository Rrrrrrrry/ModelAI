import os
import sys
current_dir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(current_dir)
from algorithms.machine_learning.ensemble.random_forest import RandomForest
from sklearn.model_selection import LeaveOneOut
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    """
    数据获取
    """
    # iris = load_iris()
    # X, y = iris.data, iris.target
    # feature_names = iris.feature_names
    # class_names = iris.target_names
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = [[0, 0], [1, 1], [2, 2], [3, 3]]
    y_train = [0, 1, 2, 3]
    X_test = [[4, 4], [5, 5]]
    feature_names = ['one', 'two']
    class_names = ['0', '1', '2', '3']

    """
    参数定义
    """
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [5, 10],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': [None],
    #     'bootstrap': [True, False]
    # }
    param_grid = {
        'n_estimators': [50],
        'criterion': ['gini'],
        'max_depth': [5],
        'min_samples_split': [2],
        'min_samples_leaf': [2],
        'max_features': [None],
        'bootstrap': [True, False]
    }

    """
    模型定义
    """
    model = RandomForest(model_type='classification', n_estimators=50)

    """
    The number of samples for each class must be less than or equal to cv
    if greater than cv, can use LeaveOneOut
    """
    cv = LeaveOneOut()
    # cv = 2
    model.grid_search_cv(X_train, y_train, param_grid=param_grid, cv=cv)
    # model.train(X_train, y_train)

    predictions = model.predict(X_test)

    print(predictions)
    print(f"feature_importances_:{model.model.feature_importances_}")

    perm_importance = permutation_importance(model.model, X_train, y_train)
    print(f"perm_importance{perm_importance.importances_mean}")
    """
    随机森林可视化（取第一棵树）
    """
    from matplotlib import pyplot as plt
    from sklearn import tree
    # 可视化整个随机森林
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(2, 2), dpi=300)
    tree.plot_tree(model.model.estimators_[0], feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()