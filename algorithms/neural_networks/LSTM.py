import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from tqdm import tqdm


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, batch_first=True):
        """
        :param input_size: 输入样本特征数量
        :param hidden_size: 每步的隐藏节点个数
        :param output_size: 输出的大小
        :param num_layers: 步长（LSTM个数）
        :param batch_first:
                如果为True:输入数据为(batch_size, seq_length, input_size),否则为(seq_length, batch_size, input_size)
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first)
        self.lr = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        output, _ = self.lstm(X)
        output = self.lr(output[:, -1, :])
        return output


def LSTM_train(input_size, hidden_size, output_size, num_layers, data_train, data_test, labels_train, labels_test,
               save_path, batch_first=True):

    """

    :param input_size: 输入大小-特征个数
    :param hidden_size: 每步的隐藏节点个数
    :param output_size: 输出大小
    :param num_layers: 步数（LSTM模块个数）
    :param data_train:训练数据
    :param data_test:测试数据
    :param labels_train:训练标签
    :param labels_test:测试标签
    :param save_path:模型保存路径
    :param batch_first:是否batch优先，
                如果为True:输入数据为(batch_size, seq_length, input_size),否则为(seq_length, batch_size, input_size)
    :return:
    """
    # 对训练集训练，保存模型，并返回测试集测试结果
    x = torch.tensor(data_train, dtype=torch.float)
    y = torch.tensor(labels_train, dtype=torch.float)

    x_t = torch.tensor(data_test, dtype=torch.float)
    y_t = torch.tensor(labels_test, dtype=torch.float)

    model = LSTM(input_size, hidden_size, output_size, num_layers=num_layers, batch_first=batch_first)
    loss_func = torch.nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=0.01)
    itr = 50
    iter_loss = []
    for i in tqdm(range(itr)):
        pred = model(x)
        loss = loss_func(pred, y)
        iter_loss.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plt.plot(range(len(iter_loss)), iter_loss, label='loss')
    plt.show()

    # 保存
    torch.save(model, save_path)

    pred = model(x_t)
    acc = loss_func(pred, y_t)
    acc = float(acc)
    print("训练测试acc", acc)
    return acc


def LSTM_test(datatest, read_path):
    """
    对测试集测试
    :param datatest:
    :param read_path:
    :return:
    """
    x_t = torch.tensor(datatest, dtype=torch.float)
    model = torch.load(read_path)
    pred = model(x_t)
    return pred.detach().numpy()