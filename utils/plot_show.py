import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# 创建示例数据

def show_voronoi(points):
    """
    计算Voronoi图并展示(泰森多边形)
    :param points:
    :return:
    """
    # 计算 Voronoi 图
    vor = Voronoi(points)

    # 输出 Voronoi 图的关键点和线段
    # print(f"Voronoi 图的顶点：\n{vor.vertices}")
    # print(f"Voronoi 图的线段连接的点的索引：\n{vor.ridge_points}")
    # print(f"Voronoi 图的线段的顶点索引：\n{vor.ridge_vertices}")
    # 绘制 Voronoi 图
    voronoi_plot_2d(vor)

    # 设置标题和标签
    plt.title('2D Voronoi Diagram')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示图形
    plt.show()


# if __name__ == '__main__':
#     points = np.random.rand(10, 2)
#     show_voronoi(points)