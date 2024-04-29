import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))
from algorithms.neural_networks.segnet import *

if __name__ == '__main__':
    # 虚构的训练数据和标签
    train_data = torch.randn(100, 3, 256, 256)  # 100个尺寸为(3, 256, 256)的图像
    train_labels = torch.randint(0, 2, (100, 256, 256))  # 100个尺寸为(256, 256)的随机标签
    val_data = torch.randn(50, 3, 256, 256)  # 验证集
    val_labels = torch.randint(0, 2, (50, 256, 256))

    test_data = torch.randn(3, 3, 256, 256)  # 验证集
    test_labels = torch.randint(0, 2, (3, 256, 256))

    # 打包并按指定批次组合到一起
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # 创建SegNet模型实例
    model = SegNet(in_channels=3, num_classes=2)  # 根据实际情况设置输入通道数和类别数

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 调用训练函数
    model_save_path = os.path.join(sys.path[0], 'model', 'best_model.pt')
    best_model_state = train_segnet(train_loader, val_loader, model, criterion, optimizer, num_epochs=2, save_path=model_save_path)
    print(f"best_model_state{best_model_state.keys()}")

    # 加载模型并测试
    loaded_model = SegNet(in_channels=3, num_classes=2)  # 重新创建一个模型实例，保持与之前相同的结构
    load_model(loaded_model, model_save_path)

    predict(test_data, test_labels, loaded_model, criterion)