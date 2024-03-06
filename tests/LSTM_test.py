import os
import sys
import torch
dir_path = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(dir_path)
from algorithms.neural_networks.LSTM import *


if __name__=='__main__':
    # 实例化 LSTM 模型
    lstm_model = LSTM(input_size=(4, 2), hidden_size=2, num_layers=2)
    print(dir(lstm_model))
    # 创建输入张量
    input_tensor = torch.randn(2, 2, 4)  # 输入维度为 (sequence_length, batch_size, input_size)
    # 前向传播获取输出
    output = lstm_model(input_tensor)

    # 打印输出张量的形状
    print("输出张量形状:", output.shape)
