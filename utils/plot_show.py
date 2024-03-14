import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# 创建示例数据

def show_voronoi(points):
    """
    计算Voronoi图并展示
    :param points:
    :return:
    """
    # 计算 Voronoi 图
    vor = Voronoi(points)

    # 绘制 Voronoi 图
    voronoi_plot_2d(vor)

    # 设置标题和标签
    plt.title('2D Voronoi Diagram')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示图形
    plt.show()
if __name__ == '__main__':
    points = np.random.rand(10, 2)
    show_voronoi(points)