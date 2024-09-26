from skimage.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

import os
import sys
import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(current_dir)
from algorithms.machine_learning.linear.linear_model import linearModel

if __name__ == '__main__':
    # 加载示例数据集

    boston = load_diabetes()
    X, y = boston.data, boston.target

    # 划分训练集和测试集
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
    model = linearModel()
    model.train(train_features, train_labels)
    y_pred = model.predict(test_features)
    mse = mean_squared_error(test_labels, y_pred)
    print(f"mse:{mse}")




