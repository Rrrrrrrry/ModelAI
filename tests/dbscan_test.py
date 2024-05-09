import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))
from algorithms.machine_learning.cluster.dbscan import *

if __name__ == '__main__':

    # 生成示例数据
    X = np.random.rand(100, 2)

    # 创建Clustering 实例
    DBSCAN_model = DBSCANClustering()

    # 执行聚类
    DBSCAN_model.fit(X)

    # 打印聚类结果
    print("簇标签：", DBSCAN_model.labels)
    # 绘制聚类结果
    plt.figure(figsize=(8, 6))
    labels = DBSCAN_model.labels
    # 绘制每个簇的数据点
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            # -1 代表噪声点
            cluster = X[labels == label]
            plt.scatter(cluster[:, 0], cluster[:, 1], color='gray', label='Noise')
        else:
            print('Cluster {}'.format(label + 1))
            cluster = X[labels == label]
            plt.scatter(cluster[:, 0], cluster[:, 1], label='Cluster {}'.format(label + 1))

    plt.title('kmaxim clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
