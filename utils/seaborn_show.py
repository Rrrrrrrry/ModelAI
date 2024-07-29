import seaborn as sns
import matplotlib.pyplot as plt


def plot_jointplot(data, x, y, kind='scatter', **kwargs):
    """
    绘制双变量关系图（Joint Plot）。

    参数：
    data : DataFrame
        数据源。
    x, y : str
        指定要绘制的两个变量的列名。
    kind : {'scatter', 'hex', 'kde'}, 可选
        指定绘图类型，默认为 'scatter'。 'scatter'（散点图，默认）、'hex'（六边形密度图）或 'kde'（核密度估计图）
    **kwargs : dict
        其他参数传递给 seaborn.jointplot() 函数。
    """
    sns.jointplot(x=x, y=y, data=data, kind=kind, **kwargs)
    plt.show()


def plot_heatmap(data, row_labels, col_labels, title='Heatmap', cmap='viridis'):
    """
    Plot a heatmap using seaborn.

    Parameters:
    - data: 2D array-like
        The data matrix to be visualized.
    - row_labels: list of str
        The labels for the rows.
    - col_labels: list of str
        The labels for the columns.
    - title: str, optional (default='Heatmap')
        The title of the heatmap.
    - cmap: str, optional (default='viridis')
        The colormap to be used for the heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, xticklabels=col_labels, yticklabels=row_labels, cmap=cmap, annot=True, fmt=".2f")
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()



if __name__ == '__main__':
    # # 示例用法
    # tips = sns.load_dataset("tips")
    # plot_jointplot(data=tips, x="total_bill", y="tip", kind="scatter")

    # 示例数据
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    row_labels = ['Row 1', 'Row 2', 'Row 3', 'Row 4']
    col_labels = ['Col 1', 'Col 2', 'Col 3', 'Col 4']

    # 绘制热力图
    plot_heatmap(data, row_labels, col_labels, title='Example Heatmap', cmap='YlGnBu')
