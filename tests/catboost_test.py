from catboost import CatBoostClassifier, Pool
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import sys
import warnings
warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(current_dir)
from algorithms.machine_learning.ensemble.catboost_model import CatBoostManager


if __name__ == '__main__':
    # 加载示例数据集
    data = load_iris()
    X, y = data.data, data.target
    # 划分训练集和测试集
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    # train_data and test_data should be CatBoost Pool objects
    train_data = Pool(data=train_features, label=train_labels)
    test_data = Pool(data=test_features, label=test_labels)

    catboost_manager = CatBoostManager(train_data, test_data)
    catboost_manager.optimize_parameters()
    # catboost_manager.train(catboost_manager.best_trial_params)
    catboost_manager.save_model(model_save_path=os.path.join(sys.path[0], "save_model", "catboost_model.cbm"))
    catboost_manager.load_model(model_load_path=os.path.join(sys.path[0], "save_model", "catboost_model.cbm"))
    pre_result = catboost_manager.predict(test_data)
    print(f"pre result: {pre_result}")
    # 获取在验证集上性能最好的迭代次数
    best_iteration = catboost_manager.model.get_best_iteration()
    # 获取在验证集上的性能指标
    print(f"catboost_manager:{catboost_manager}")
    print(f"catboost_manager.model:{catboost_manager.model}")
    eval_metrics = catboost_manager.model.get_evals_result()
    print(f"best_iteration{best_iteration}")
    print(f"eval_metrics{eval_metrics}")
    # catboost_manager.test()
