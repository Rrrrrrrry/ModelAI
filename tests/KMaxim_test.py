# 示例用法
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
current_dir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(current_dir)
from algorithms.machine_learning.cluster.KMaxim import KMaximClustering

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(100, 2)
    kmaxim = KMaximClustering(k=3)
    kmaxim.fit(X)
    print("Cluster labels:", kmaxim.labels)

    # 绘制聚类结果
    plt.figure(figsize=(8, 6))
    labels = kmaxim.labels
    # 绘制每个簇的数据点
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            # -1 代表噪声点
            cluster = X[labels == label]
            plt.scatter(cluster[:, 0], cluster[:, 1], color='gray', label='Noise')
        else:
            cluster = X[labels == label]
            plt.scatter(cluster[:, 0], cluster[:, 1], label='Cluster {}'.format(label + 1))

    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
