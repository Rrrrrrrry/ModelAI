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


# if __name__ == '__main__':
#     # 示例用法
#     tips = sns.load_dataset("tips")
#     plot_jointplot(data=tips, x="total_bill", y="tip", kind="scatter")
