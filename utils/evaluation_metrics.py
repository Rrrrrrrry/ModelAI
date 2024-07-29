import numpy as np
from typing import Callable
from scipy.integrate import simps


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import numpy as np
from matplotlib import pyplot as plt
"""
评价指标
"""

def cal_mse_point(signal1, signal2):
    """
    点对点计算均方误差mse，越小越好
    :param signal1:
    :param signal2:
    :return:
    """
    # 计算信号1和信号2的差的平方
    squared_difference = (signal1 - signal2) ** 2
    # 计算均方误差
    mse = np.mean(squared_difference)
    return mse


def cal_mse(signal1, signal2):
    """
    计算均方误差mse，越小越好
    :param signal1:
    :param signal2:
    :return:
    """
    mse = abs(np.mean(signal1) - np.mean(signal2))
    return mse


def cal_min(signal1, signal2):
    """
    计算均方误差mse，越小越好
    :param signal1:
    :param signal2:
    :return:
    """
    min_dif = abs(np.min(signal1) - np.min(signal2))
    return min_dif


def smoothness_evaluation(data, predict: Callable = None):
    """
    判断曲线的平滑性，计算曲线的梯度变化率
    predict:
    :param data: 曲线数据
    :return: 平滑性评估值
    """
    # 计算曲线的梯度
    if isinstance(predict, Callable):
        data = predict(data)
    gradients = np.gradient(data)
    # 计算梯度的变化率，这里简单地计算梯度的标准差作为平滑性评估值
    smoothness = np.std(gradients)
    # smoothness = np.nanmean(gradients)
    # print(f"smoothness{smoothness}")
    return smoothness


def compute_bending_energy(y):
    """
    弯曲能量（基于曲线的弯曲率的平方积分）衡量平滑度，越小越好
    曲率=(x'y" - x"y')/((x')^2 + (y')^2)^(3/2)
    :param y:
    :return:
    """
    x = np.arange(len(y))
    # 计算参数方程的一阶导数
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)

    # 计算参数方程的二阶导数
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    # 计算曲率的平方
    curvature_squared = (dx_dt*d2y_dt2 - dy_dt*d2x_dt2)**2 / (dx_dt**2 + dy_dt**2)**3

    # 计算弯曲能量
    bending_energy = simps(curvature_squared)

    return bending_energy



def plot_multiclass_roc(y_true, y_score, classes, title='Receiver operating characteristic', map_dict=None):
    """
    绘制多分类任务中的ROC曲线
    :param y_true: 真实标签(n_samples,) or (n_samples, n_classes)
    :param y_score: 预测概率标签(n_samples, n_classes)
    :param classes: 预测的类别列表 (n_classes,)
    :param title:标题
    :param map_dict:类别映射
    :return:
    """
    n_classes = len(classes)
    if y_true.ndim == 1:
        y_true = label_binarize(y_true, classes=range(n_classes))
        if n_classes == 2:
            y_true = np.hstack((1 - y_true, y_true))
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
    for i, color in zip(range(n_classes), colors):
        class_name = classes[i]
        if map_dict is not None and class_name in map_dict:
            class_name = map_dict[class_name]
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_name, roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))

    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=4,
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()



def plot_binary_roc(y_true, y_score, title='Receiver Operating Characteristic'):
    """
    绘制二分类的roc曲线
    :param y_true:真实标签(n_samples,)
    :param y_score:预测标签（1类）(n_samples,)
    :param title:标题
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {0:0.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

#
# if __name__ == '__main__':
#     # 示例数据
#     from sklearn.datasets import make_classification
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import label_binarize
#     from sklearn.svm import SVC
#     from sklearn.multiclass import OneVsRestClassifier
#
#     classes = [0, 3]
#     # 生成一个示例数据集
#     X, y = make_classification(n_samples=1000, n_features=20, n_classes=len(classes), n_informative=3, n_clusters_per_class=1)
#
#     # 数据集分割
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
#
#     # 使用One-vs-Rest分类器和SVM
#     classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=0))
#     y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
#     # 绘制ROC曲线
#     if len(classes) == 2:
#         plot_binary_roc(y_test, y_score[:, 1])
#     plot_multiclass_roc(y_test, y_score, classes)
#
