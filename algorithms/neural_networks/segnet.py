import os.path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm

class SegNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegNet, self).__init__()
        self.encoder_conv_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder_conv_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_conv_4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.encoder_conv_5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.decoder_conv_5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.decoder_conv_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.decoder_conv_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv_1 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.encoder_conv_1(x))
        x1, idxs1 = self.pool(x1)

        x2 = F.relu(self.encoder_conv_2(x1))
        x2, idxs2 = self.pool(x2)

        x3 = F.relu(self.encoder_conv_3(x2))
        x3, idxs3 = self.pool(x3)

        x4 = F.relu(self.encoder_conv_4(x3))
        x4, idxs4 = self.pool(x4)

        x5 = F.relu(self.encoder_conv_5(x4))
        x5, idxs5 = self.pool(x5)

        # Decoder
        x5 = self.unpool(x5, idxs5)
        x5 = F.relu(self.decoder_conv_5(x5))

        x4 = self.unpool(x5, idxs4)
        x4 = F.relu(self.decoder_conv_4(x4))

        x3 = self.unpool(x4, idxs3)
        x3 = F.relu(self.decoder_conv_3(x3))

        x2 = self.unpool(x3, idxs2)
        x2 = F.relu(self.decoder_conv_2(x2))

        x1 = self.unpool(x2, idxs1)
        x1 = F.relu(self.decoder_conv_1(x1))

        return x1


def train_segnet(train_loader, val_loader, model, criterion, optimizer, num_epochs=10, save_path=None):
    """
    训练SegNet模型

    参数:
        train_loader (torch.utils.data.DataLoader): 用于加载训练数据和标签的数据加载器
        # train_data (torch.Tensor): 训练数据集，形状为 (N, C, H, W) 其中 N 是样本数，C 是通道数，H 和 W 是图像高度和宽度
        # train_labels (torch.Tensor): 训练标签，形状为 (N, H, W)，其中 N 是样本数，H 和 W 是图像高度和宽度
        model (torch.nn.Module): SegNet模型实例
        criterion (torch.nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        num_epochs (int): 训练的epoch数量，默认为10
    """
    best_val_loss = np.inf
    best_model_state = None
    for epoch in tqdm(range(num_epochs)):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        # for inputs, labels in zip(train_data, train_labels):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # outputs = model(inputs.unsqueeze(0))  # 增加一个维度以匹配模型输入的形状
            # loss = criterion(outputs, labels.unsqueeze(0))  # 增加一个维度以匹配标签的形状
            outputs = model(inputs)  # 增加一个维度以匹配模型输入的形状
            loss = criterion(outputs, labels)  # 增加一个维度以匹配标签的形状
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证模型
        model.eval()    # 将模型设置为评估模式
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # 计算平均损失
        running_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss}, Val Loss: {val_loss}')
        # 保存最佳模型状态
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
    print('Finished Training')
    if save_path is not None:
        print(os.path.dirname(save_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model_state, save_path)
    return best_model_state


def load_model(model, model_path):
    """
    加载模型参数状态字典

    参数:
        model (torch.nn.Module): 模型实例
        model_path (str): 模型参数状态字典的文件路径
    """
    model.load_state_dict(torch.load(model_path))
    print(f'Model loaded from {model_path}')


def predict(test_data, test_labels, model, criterion):
    """
    使用测试数据集对模型进行测试，并返回测试损失和其他指标

    参数:
        test_loader (torch.utils.data.DataLoader): 用于加载测试数据和标签的数据加载器
        model (torch.nn.Module): 已加载的模型实例
        criterion (torch.nn.Module): 损失函数
    返回:
        test_loss (float): 测试损失
        other_metrics: 其他指标，根据需要定义
    """
    model.eval()  # 设置模型为评估模式
    test_loss = 0.0
    # 其他指标的计算

    with torch.no_grad():
        for inputs, labels in zip(test_data, test_labels):
            outputs = model(inputs.unsqueeze(0))
            print(f"outputs->{outputs.shape}")
            predicted_class = torch.argmax(outputs, dim=1)
            print(f"predicted_class->{predicted_class.shape}")
            loss = criterion(outputs, labels.unsqueeze(0))
            test_loss += loss.item()
            # 创建预测结果图像
            # predicted_class = np.argmax(predicted_class, axis=0)
            # plt.figure(figsize=(6, 6))
            # plt.imshow(predicted_class, cmap='jet')  # 使用jet colormap来表示不同类别
            # plt.colorbar()
            # plt.title(f'Predicted Segmentation for Sample')
            # plt.show()

            # 其他指标的计算

    test_loss /= len(test_data)
    # 其他指标的计算

    print(f'Test Loss: {test_loss}')
    # 打印其他指标

    return test_loss