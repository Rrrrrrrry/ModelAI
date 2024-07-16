import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

def plot_3d_data(df, x_col, y_col, z_col, color_by_list, colormap='viridis'):
    """
    生成3D散点图，显示X、Y、Z数据，并按color_by_list着色。
    :param df: 包含井数据的Pandas DataFrame。
    :param x_col: x坐标列名
    :param y_col: y坐标列名
    :param z_col: z坐标列名
    :param color_by: [one_col, two_col, three_col]，如果有一列，则按这一列，如果有两列，则按两列
    :param colormap: Matplotlib颜色映射名称。
    :return:
    """
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 根据color_by参数选择颜色编码方式
    color_by = '_'.join(color_by_list)
    if len(color_by_list) > 1:
        df[color_by] = df[color_by_list].apply(lambda row: '_'.join(row.astype(str)), axis=1)
    colors = df[color_by].values

    # 为每个不同的颜色分配一个唯一的颜色
    unique_colors = np.unique(colors)
    color_map = plt.get_cmap(colormap)
    color_array = [color_map(i / len(unique_colors)) for i in range(len(unique_colors))]
    color_dict = dict(zip(unique_colors, color_array))

    # 绘制散点图
    for color in unique_colors:
        subset = df[df[color_by].apply(lambda x: str(x) if isinstance(x, int) else x) == str(color)]
        ax.scatter(subset[x_col], subset[y_col], subset[z_col], c=color_dict[color], label=str(color), s=60)

    # 设置图表标题和坐标轴标签
    ax.set_title('3D Visualization of Well Data')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)

    # 添加图例
    handles, labels = [], []
    for l, c in color_dict.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10))
        labels.append(l)
    ax.legend(handles, labels, loc='upper right')

    plt.show()


# if __name__ == '__main__':
#     # points = np.random.rand(10, 2)
#     # show_voronoi(points)
#
#     data = pd.DataFrame({'X':[1, 2, 3, 5], 'Y':[1, 5, 3, 6], 'Z':[2, 4, 1, 6], 'name':['a', 'a', 'b', 'b'], 'id':['s', 's', 's', 'c']})
#     plot_3d_data(data, 'X', 'Y', 'Z', color_by_list=['name', 'id'])