from catboost import CatBoostClassifier, Pool
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import sys
import warnings
warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(current_dir)
from algorithms.machine_learning.ensemble.lightGBM_model import LightGBM_Manager
from sklearn.preprocessing import LabelEncoder
from utils.model_utils import *

if __name__ == '__main__':
    # 加载示例数据集
    data = load_iris()
    X, y = data.data, data.target
    # 确保 y 是整数类型
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # 划分训练集和测试集
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    LightGBM_m = LightGBM_Manager(train_features, train_labels, test_features, test_labels)
    LightGBM_m.optimize_parameters()
    LightGBM_m.train(LightGBM_m.best_trial_params)
    save_model(LightGBM_m, os.path.join(sys.path[0], 'save_model/LightGBM_model.joblib'))
    LightGBM_m = load_model(os.path.join(sys.path[0], 'save_model/LightGBM_model.joblib'), data_format='joblib')
    pre_result = LightGBM_m.predict(test_features)
    print(f"pre result: {pre_result}")
    # # 获取在验证集上性能最好的迭代次数
    best_iteration = LightGBM_m.model._best_iteration
    # # 获取在验证集上的性能指标
    print(f"catboost_manager:{LightGBM_m}")
    print(f"catboost_manager.model:{LightGBM_m.model}")
    eval_metrics = LightGBM_m.model._evals_result
    print(f"best_iteration{best_iteration}")
    print(f"eval_metrics{eval_metrics}")
    # catboost_manager.test()
