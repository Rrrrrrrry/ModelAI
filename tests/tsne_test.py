import os.path
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

current_dir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(current_dir)
from algorithms.machine_learning.feature_extraction.TSNE import *

if __name__ == '__main__':
    # 加载手写数字数据集
    digits = load_digits()
    X = digits.data
    y = digits.target
    x_tsne = tsne(X, n_components=2, random_state=42)

    # 可视化降维后的数据
    plt.figure(figsize=(10, 8))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10), marker='o', edgecolor='none', alpha=0.6)
    plt.colorbar(label='digit label', ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.title('t-SNE visualization of hand-written digits dataset')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()
