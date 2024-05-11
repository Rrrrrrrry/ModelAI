import numpy as np
from typing import Callable
from scipy.integrate import simps


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



