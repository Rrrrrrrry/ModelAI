import os
import sys

current_dir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(current_dir)
from algorithms.machine_learning.ensemble.random_forest import RandomForest
from sklearn.model_selection import LeaveOneOut
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    model = RandomForest(model_type='classification')

    X = [[0, 0], [1, 1], [2, 2], [3, 3]]
    y = [0, 1, 2, 3]

    param_grid = {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None],
        'bootstrap': [True, False]
    }

    """
    The number of samples for each class must be less than or equal to cv
    if greater than cv, can use LeaveOneOut
    """
    cv = LeaveOneOut()
    # cv = 2
    model.train(X, y, param_grid=None, cv=cv)

    predictions = model.predict([[4, 4], [5, 5]])

    print(predictions)
    print(f"feature_importances_:{model.model.feature_importances_}")

    perm_importance = permutation_importance(model.model, X, y)
    print(f"perm_importance{perm_importance.importances_mean}")

    from matplotlib import pyplot as plt
    from sklearn import tree
    # 可视化整个随机森林
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=300)
    tree.plot_tree(model.model.estimators_[0], feature_names=['one', 'two'], class_names=['0', '1', '2', '3'], filled=True)
    plt.show()