import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))
from algorithms.machine_learning.cluster.Kmeans import *

if __name__ == '__main__':

    # 生成示例数据
    data = np.random.rand(100, 2)

    # 创建 KMeansClustering 实例
    kmeans_model = KMeansClustering(n_clusters=3)

    # 执行聚类
    kmeans_model.fit(data)

    # 打印聚类结果
    print("聚类中心点：", kmeans_model.centers)
    print("簇标签：", kmeans_model.labels)
    vor = Voronoi(kmeans_model.centers)
    # 绘制 Voronoi 图
    voronoi_plot_2d(vor)
    # 绘制原始图像和中心点
    plt.scatter(data[:, 0], data[:, 1], c=kmeans_model.labels, cmap='viridis', alpha=0.5)
    plt.scatter(kmeans_model.centers[:, 0], kmeans_model.centers[:, 1], c='red', marker='x')
    plt.show()
