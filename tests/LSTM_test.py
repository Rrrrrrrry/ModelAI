import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))
from algorithms.neural_networks.LSTM import *

if __name__ == '__main__':
    """
    目标：
    1000条数据，特征数为5，用LSTM，4步预测3步，包含数据集的切分、模型训练、测试，torch实现。
    # model = LSTM(input_size=5, hidden_size=2, output_size=3, num_layers=4, batch_first=True)
    # output = model(X)
    # print(X.shape)
    # print(output.shape)
    """
    """
    参数定义
    """
    input_size, hidden_size, output_size, num_layers = 5, 2, 3, 4

    """
    构造数据集，示例
    """
    data = np.random.randn(1000, 5)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)

    use_step = 4
    pre_step = 3
    final_x_data = []  # (994, 4, 5)
    final_y_data = []  # (994, 3)
    for i in range(0, len(data) - pre_step - use_step + 1, 1):
        final_x_data.append(data[i:i + use_step])
        final_y_data.append(data[i + use_step:i + use_step + pre_step])
    final_x_data = np.array(final_x_data)
    final_y_data = np.array(final_y_data)[:, :, -1]
    X_train, X_test, y_train, y_test = train_test_split(final_x_data, final_y_data, test_size=0.3, random_state=42)

    """
    训练模型
    """
    save_path = os.path.join(sys.path[0], 'save_model', 'lstm.pt')
    # LSTM_train(input_size, hidden_size, output_size, num_layers, X_train, X_test, y_train, y_test, save_path,
    #            batch_first=True)

    """
    测试模型
    """
    pre_t = LSTM_test(X_test, save_path)
    print(f"pre_t{pre_t.shape}")