import os
import sys

from torch_geometric.datasets import Planetoid

current_dir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(current_dir)
from algorithms.neural_networks.GCN import *
from torch_geometric.transforms import NormalizeFeatures
from algorithms.machine_learning.feature_extraction.TSNE import *
import matplotlib.pyplot as plt


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


if __name__ == '__main__':
    # 获取数据
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())  # transform预处理
    # 训练模型
    save_path = os.path.join(sys.path[0], 'save_model', 'gcn.pt')
    # train_model(dataset, epochs=101, lr=0.01, save_path=save_path)
    # 测试数据
    model = torch.load(save_path)
    out, test_acc = predict(model, dataset)
    print(test_acc)

    model.eval()
    out_show = model(dataset.x, dataset.edge_index)
    print(f"out_show{out_show}")
    visualize(out_show, color=dataset.y)
