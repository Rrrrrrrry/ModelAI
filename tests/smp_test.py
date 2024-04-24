"""
自定义数据集测试图像分割网络，不够全面，仅测试版本
"""
import os.path
import sys
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from PIL import Image
import numpy as np
import cv2
from typing import List

current_dir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(current_dir)
from utils.file_op import *
from utils.model_utils import *
from algorithms.neural_networks.smp_model import *


def get_dataloader(dataset, batch_size=8, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def resize_with_label(image, label, new_size, interpolation=cv2.INTER_NEAREST, multiple_num=32):
    # 需要调整到32的倍数
    # # 计算宽度和高度的32的倍数
    # width, height, _ = image.shape
    # new_width = ((width - 1) // multiple_num + 1) * multiple_num
    # new_height = ((height - 1) // multiple_num + 1) * multiple_num
    # new_size = [new_width, new_height]
    # 调整图像大小
    resized_image = cv2.resize(image, new_size, interpolation=interpolation)

    # 如果标签是二维数组，与图像具有相同的形状
    if len(label.shape) == 2:
        resized_label = cv2.resize(label, new_size, interpolation=interpolation)
    # 如果标签是一维数组，进行相应的调整
    else:
        # 计算调整比例
        ratio_x = new_size[1] / image.shape[1]
        ratio_y = new_size[0] / image.shape[0]
        # 根据比例调整标签位置
        resized_label = label * [ratio_x, ratio_y]

    return resized_image, resized_label


class MyDataset(Dataset):
    def __init__(self, x):
        self.x: List[dict] = list(x.values())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = self.x[idx]["data"]
        labels = self.x[idx]['label']

        data_image = np.array(Image.open(data).convert('RGB'))
        labels_image = np.array(Image.open(labels))
        # 调整图像大小和标签位置
        data_image, labels_image = resize_with_label(data_image, labels_image, [224, 224])
        data_tensor = torch.tensor(data_image, dtype=torch.float).permute(2, 0, 1)
        labels_tensor = torch.tensor(labels_image, dtype=torch.long)
        return data_tensor, labels_tensor


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=2):
    model.to(device)
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        training_bar = tqdm(train_loader, desc='Training Progress', position=0)
        for images, masks in training_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 更新进度条的动态信息
            training_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     num_epochs,
                                                                     loss)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        epoch_losses.append(epoch_loss)
    print(f"epoch_losses:{epoch_losses}")
    return epoch_losses


def visualize(model, train_loader, device):
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        images, masks = sample_batch
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        pred_masks = torch.argmax(outputs, dim=1)
        grid_image = make_grid(images.cpu(), nrow=4)
        grid_pred_mask = make_grid(pred_masks.unsqueeze(1).cpu(), nrow=4)
        grid_true_mask = make_grid(masks.unsqueeze(1).cpu(), nrow=4)

    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.imshow(grid_image.permute(1, 2, 0))
    plt.title('输入图像')
    plt.subplot(3, 1, 2)
    plt.imshow(grid_true_mask.permute(1, 2, 0), cmap='jet')
    plt.title('真实掩码')
    plt.subplot(3, 1, 3)
    plt.imshow(grid_pred_mask.permute(1, 2, 0), cmap='jet')
    plt.title('预测掩码')
    plt.show()


def evaluate_model(model, test_loader, device):
    model.eval()
    pixel_accuracy = 0.0
    class_precision = 0.0
    intersection = 0
    union = 0

    with torch.no_grad():
        for images, masks in test_loader:
            print(f"images: {images.shape}")
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)

            # 计算像素准确率
            pixel_accuracy += accuracy_score(masks.view(-1).cpu(), pred_masks.view(-1).cpu())

            # 计算混淆矩阵
            cm = confusion_matrix(masks.view(-1).cpu(), pred_masks.view(-1).cpu())
            intersection += np.diag(cm).sum()
            union += (cm.sum(1) + cm.sum(0) - np.diag(cm)).sum()

    # 计算平均像素准确率
    pixel_accuracy /= len(test_loader)

    # 计算IoU
    iou = intersection / union
    return pixel_accuracy, iou


def train_process(dataloader, save_path=None, model_name='mobilenet_v2', num_epochs=100):
    # 定义模型
    if model_name == 'unet':
        model = smp_unet()
    elif model_name == 'pspnet':
        model = smp_pspnet()
    else:
        raise ValueError(f"model_name must in {'unet', 'pspnet'}")
    # else:
    #     model = UNetWithSPP(in_channels=3, out_channels=6, pool_sizes=[(1,1), (2,2), (3,3)])
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_losses_array = train_model(model, dataloader, criterion, optimizer, device, num_epochs=num_epochs)
    if save_path is not None:
        save_model(model, os.path.join(save_path, f'{model_name}.joblib'), data_format='joblib')
        save_npy(epoch_losses_array, os.path.join(save_path, f'{model_name}_epoch_losses.npy'))


def predict_process(dataloader, save_path=None, model_name='mobilenet_v2'):
    """
    测试
    :param dataloader:
    :param save_path:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(os.path.join(save_path, f'{model_name}.joblib'), data_format='joblib')
    pixel_accuracy, iou = evaluate_model(model, dataloader, device)
    print(f"pixel_accuracy, iou:{pixel_accuracy, iou}")


def get_path_dict(dataset_path, label_path):
    path_dict = {}
    for image in os.listdir(dataset_path):
        name = image.split(".")[0]
        path_dict.update({name: {"data": os.path.join(dataset_path, image),
                                 "label": os.path.join(label_path, name + ".png")}}
                         )
    return path_dict


if __name__ == '__main__':
    """
    测试版-仅用于训练测试，没完善更多功能
    """
    # 加载训练数据
    root_path = os.path.join(sys.path[0], 'data', 'UDD6', 'train')
    train_dataset = os.path.join(root_path, 'images')
    labels = os.path.join(root_path, 'masks')
    path_dict = get_path_dict(train_dataset, labels)
    dataloader = MyDataset(path_dict)
    train_loader = get_dataloader(dataloader, batch_size=8)
    # 加载测试数据
    root_path = os.path.join(sys.path[0], 'data', 'UDD6', 'val')
    train_dataset = os.path.join(root_path, 'images')
    labels = os.path.join(root_path, 'masks')
    path_dict = get_path_dict(train_dataset, labels)
    dataloader = MyDataset(path_dict)
    val_loader = get_dataloader(dataloader, batch_size=1)

    model_save_path = os.path.join(sys.path[0], 'models_save')
    os.makedirs(model_save_path, exist_ok=True)

    # unet/pspnet/
    model_name = 'unet'
    train_process(train_loader, save_path=model_save_path, model_name=model_name, num_epochs=2)
    # 100代：pixel_accuracy, iou:(0.815849922558309, 0.6889751038322104)
    predict_process(val_loader, save_path=model_save_path, model_name=model_name)
    epoch_losses = load_npy(os.path.join(model_save_path, f'{model_name}_epoch_losses.npy'))
    print(f"epoch_losses{epoch_losses}")

    # 加载 PNG 图像
    # image_path = r"D:\python_program\RomentSegment\datasets\UDD6\train\masks/000001.png"
    # image = Image.open(image_path)
    #
    # # 将图像转换为 NumPy 数组
    # image_array = np.array(image)
    #
    # print(image_array.shape)
    # print(image_array[0][0:5])
