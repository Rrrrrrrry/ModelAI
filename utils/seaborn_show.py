import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
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
    sns.heatmap(data, xticklabels=row_labels, yticklabels=col_labels, cmap=cmap, annot=True, fmt=".2f")
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()


def plot_confusion_matrix(cm, row_labels, col_labels,
                          normalize=False,
                          title='confusion_matrix',
                          cmap=plt.cm.Blues,
                          filename='confusion_matrix.png', if_save=False, if_show=False, fonts_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    The resulting plot is saved to a file specified by `filename`.
    """
    from sklearn.metrics import roc_curve, auc

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('混淆矩阵')

    # plt.figure(figsize=(8, 6))
    plt.figure(figsize=(15, 12))
    # sns.set(style="white")
    sns.set_color_codes(palette='deep')
    ax = sns.heatmap(cm, annot=True, annot_kws={"size": 16}, cmap=cmap,
                     xticklabels=row_labels, yticklabels=col_labels)

    ax.set_title(title, fontproperties=font_manager.FontProperties(fname=fonts_path))
    ax.set_xlabel('Predicted label', fontproperties=font_manager.FontProperties(fname=fonts_path))
    ax.set_ylabel('True label', fontproperties=font_manager.FontProperties(fname=fonts_path))
    ax.xaxis.set_ticklabels(row_labels, fontproperties=font_manager.FontProperties(fname=fonts_path))
    ax.yaxis.set_ticklabels(col_labels, fontproperties=font_manager.FontProperties(fname=fonts_path))
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.tight_layout()
    print(f"filename{filename}")
    if if_save:
        print('save')
        plt.savefig(filename)
        plt.close()  # Close the plot to avoid memory leaks
    if if_show:
        plt.show()


def plot_kde(data, label=None, ax=None, **kwargs):
    """
    绘制核密度估计图。

    :param data: 数据序列。
    :param label: 图例标签。
    :param ax: Matplotlib Axes 对象。
    :param kwargs: 其他传递给 sns.kdeplot 的关键字参数。
    """
    if ax is None:
        ax = plt.gca()

    sns.kdeplot(data=data, fill=True, label=label, ax=ax, **kwargs)
    plt.legend()  # 显示图例
    plt.show()

if __name__ == '__main__':
    # # 示例用法
    # tips = sns.load_dataset("tips")
    # plot_jointplot(data=tips, x="total_bill", y="tip", kind="scatter")

    # # 示例数据
    # data = [
    #     [1, 2, 3, 4],
    #     [5, 6, 7, 8],
    #     [9, 10, 11, 12],
    #     [13, 14, 15, 16]
    # ]
    #
    # row_labels = ['行 1', 'Row 2', 'Row 3', 'Row 4']
    # col_labels = ['Col 1', '列 2', 'Col 3', 'Col 4']
    #
    # # 绘制热力图
    # plot_heatmap(data, row_labels, col_labels, title='Example Heatmap', cmap='YlGnBu')
    #
    # plot_confusion_matrix(data, row_labels, col_labels,
    #                           normalize=False,
    #                           title='矩阵',
    #                           cmap=plt.cm.Blues,
    #                           filename='confusion_matrix.png', if_save=False, if_show=True)

    # 绘制核密度估计图
    d = np.random.normal(size=100)  # 生成一些随机数据
    plot_kde(d, label="Example Data")

