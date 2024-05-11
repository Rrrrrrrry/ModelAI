import copy
import logging

import numpy as np
import pywt
from PyEMD import EMD, EEMD
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import os
import sys
import scipy.signal as sig
from scipy.signal import argrelextrema, savgol_filter
from scipy.interpolate import interp1d

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0])))
sys.path.append(current_dir)
from utils.evaluation_metrics import compute_bending_energy
# from scripts.data_preprocess import del_outliers, edge_padding
from algorithms.machine_learning.feature_extraction.data_preprocess import *

class BaseModel(BaseEstimator):
    def fit(self, data: np.ndarray, **kwargs) -> np.ndarray:
        raise


class MVFilter(BaseModel):
    def __init__(self, window_size=8, outlier_k=None, *kargs, **kwargs):
        """
        移动平均滤波
        :param window_size:平滑窗口大小
        :param outlier_k:k-sigma方式去除窗口内的异常值
        :return:
        """
        self.window_size = window_size
        self.outlier_k = outlier_k

    def fit(self, signal: np.ndarray, **kwargs):
        filtered_signal = np.zeros_like(signal)  # 创建一个与输入信号相同大小的零数组
        half_window = self.window_size // 2
        for i in range(0, len(signal)):
            # 计算滑窗内的数据的平均值
            window = signal[max(i - half_window, 0): i + half_window + 1]
            if len(window) == 0:
                filtered_signal[i] = signal[i]
            else:
                # 删除窗口内的异常点
                window_del_outliers = []
                if self.outlier_k is not None:
                    window_del_outliers = sigma_del_outliers(window, k=self.outlier_k)
                if len(window_del_outliers) > 0:
                    window = window_del_outliers
                window_average = np.mean(window)
                # 将平均值放入滤波后的信号数组中
                filtered_signal[i] = window_average

        # 处理边界数据
        # filtered_signal[:half_window] = signal[:half_window]
        # filtered_signal[-half_window:] = signal[-half_window:]

        return filtered_signal

    def predict(self, x):
        return self.fit(x)


class WaveFilter(BaseModel):
    def __init__(self, wavelet='sym8', level=4, threshold_method='greater', mode='symmetric', *kargs, **kwargs):
        """
        对数据进行小波变换
        :param signal: 需要小波变换的数据
        :param wavelet: 小波类型    'sym8'
        :param level: 分解级数 4
        :param threshold_method:阈值方法 soft
        :return:
        """
        self.wavelet = wavelet
        self.level = level
        self.threshold_method = threshold_method
        self.mode = mode
        # super().__init__()

    @staticmethod
    def median_absolute_deviation(data):
        """
        中位数绝对偏差（MAD）
        :param data:
        :return:
        """
        median = np.median(data)
        deviations = np.abs(data - median)
        mad = np.median(deviations)
        return mad

    def estimated_noise_std(self, data):
        """
        获取估计噪声标准层sigma，用于后续计算阈值
        :param data:
        :return:
        """
        # 中位绝对方差
        sigma = self.median_absolute_deviation(data) / 0.6745
        return sigma

    def fit(self, signal: np.ndarray, y=None, **kwargs):
        signal = np.array(signal)
        # 进行小波变换
        wave_coeffs = pywt.wavedec(signal, self.wavelet, level=self.level, mode=self.mode)

        # 获取估计噪声标准差
        sigma = self.estimated_noise_std(signal)

        noise_std = sigma * np.sqrt(2 * np.log(len(signal)))

        # 使用特定的阈值方法进行小波系数的阈值去噪
        denoised_coeffs = []
        for i, coeff in enumerate(wave_coeffs[:]):
            # 计算阈值
            t = noise_std * np.sqrt(2 * np.log(len(coeff)))
            threshold = pywt.threshold(coeff, value=t, mode=self.threshold_method)
            denoised_coeffs.append(threshold)

        # 重构信号
        reconstructed_signal = pywt.waverec(denoised_coeffs, self.wavelet, mode=self.mode)
        return reconstructed_signal

    def predict(self, x):
        return self.fit(x)


class EmdFilter(BaseModel):
    def __init__(self, max_num_imf=4, choose_level=-1, *args, **kwargs):
        """
        EMD经验模态分解
        :param signal:需要进行emd变换的数据
        :param max_num_imf:最大的 IMF 数量
        :return:
        """
        self.max_num_imf = max_num_imf
        self.choose_level = choose_level

    def fit(self, signal: np.ndarray, y=None, **kwargs):
        signal = np.array(signal)
        emd = EMD()
        # 设置 EMD 参数
        emd.MAX_IMF = self.max_num_imf  # 最大的 IMF 数量
        emd.FIXE_H = 0  # 不使用固定阈值
        emd.FIXE = 0  # 不使用固定频率
        emd.SIFT = 0  # 不使用 SIFT 过程
        emd_result = emd(signal)[self.choose_level]
        return emd_result

    def predict(self, x):
        return self.fit(x)


class EEmdFilter(BaseModel):
    def __init__(self, max_num_imf=4, choose_level=-1, *args, **kwargs):
        """
        EMD经验模态分解
        :param signal:需要进行emd变换的数据
        :param max_num_imf:最大的 IMF 数量
        :return:
        """
        self.max_num_imf = max_num_imf
        self.choose_level = choose_level

    def fit(self, signal: np.ndarray, y=None, **kwargs):
        signal = np.array(signal)
        # 初始化EEMD对象
        eemd = EEMD()
        eemd.MAX_IMF = self.max_num_imf  # 最大的 IMF 数量
        eemd.FIXE_H = 0  # 不使用固定阈值
        eemd.FIXE = 0  # 不使用固定频率
        eemd.SIFT = 0  # 不使用 SIFT 过程
        # 执行EEMD分解
        emd_result = eemd.eemd(signal, np.array(range(len(signal))))[self.choose_level]
        return emd_result

    def predict(self, x):
        return self.fit(x)


class STFTFilter(BaseModel):
    def __init__(self, window_size=256, noverlap=None, threshold=0.5, window='hann', boundary='even', *args, **kwargs):
        """
        短时傅里叶变换
        :param window_size:每个段的样本数
        :param noverlap:段之间重叠点的数目
        :param threshold:去噪阈值
        :param window:使用的窗函数
        :return:
        """
        self.window_size = window_size
        self.noverlap = noverlap
        self.threshold = threshold
        self.window = window
        self.boundary = boundary

    def fit(self, signal: np.ndarray, **kwargs):
        """
        短时傅里叶变换
        :return:
        """
        signal = np.array(signal)
        # 窗口大小不能超过原始数据的大小
        self.window_size = min(self.window_size, len(signal))
        # 计算STFT
        f, t, Zxx = sig.stft(signal, window=self.window, nperseg=self.window_size, noverlap=self.noverlap,
                             boundary=self.boundary)

        # 应用阈值，将低于阈值的幅度置为0
        Zxx[np.abs(Zxx) < self.threshold] = 0

        # 逆STFT恢复信号
        _, filtered_signal = sig.istft(Zxx, window=self.window, nperseg=self.window_size, noverlap=self.noverlap)
        filtered_signal = filtered_signal[0:len(signal)]

        return filtered_signal

    def predict(self, x):
        return self.fit(x)


class ExtremumFilter(BaseModel):
    def __init__(self, segment_num=None, *args, **kwargs):
        """
        分段极值滤波
        :param segment_num:分段极值滤波对应的每段包含的样本点个数
        :param noverlap:段之间重叠点的数目
        :param threshold:去噪阈值
        :param window:使用的窗函数
        :return:
        """
        self.segment_num = segment_num

    @staticmethod
    def get_cal_smothness(y):
        data_minmax = MinMaxScaler().fit_transform(y.copy().reshape(-1, 1)).flatten()
        data_smothness = compute_bending_energy(data_minmax)
        return data_smothness

    def data_interpolation_greater_less(self, x, y, maxima_x, maxima_y, minima_x, minima_y):
        """
        根据获取到的极大值和极小值点，插值出极值中线
        :param x:原始数据x
        :param y:原始数据y
        :param maxima_x:极大值x坐标
        :param maxima_y:极大值y坐标
        :param minima_x:极小值x坐标
        :param minima_y:极小值y坐标
        :return:
        """
        interpolation_x = x
        interpolation_y = y
        if len(maxima_x) > 1 and len(maxima_y) > 1:
            # 构造样条插值线段
            maxima_interpolation = interp1d(maxima_x, maxima_y, kind='linear', fill_value="extrapolate")
            minima_interpolation = interp1d(minima_x, minima_y, kind='linear', fill_value="extrapolate")

            # 计算插值结果
            maxima_interpolation_y = maxima_interpolation(interpolation_x)
            minima_interpolation_y = minima_interpolation(interpolation_x)

            # interpolation_y = (maxima_interpolation_y + minima_interpolation_y) / 2
            interpolation_y = maxima_interpolation_y - abs(maxima_interpolation_y - minima_interpolation_y) / 3
            # 240223
            # interpolation_y = MVFilter(20, outlier_k=3).predict(interpolation_y)

            # import matplotlib.pyplot as plt
            # plt.plot(x, y, label='extrema')
            # plt.plot(interpolation_x, maxima_interpolation_y, label='maxima_interpolation_y')
            # plt.plot(interpolation_x, minima_interpolation_y, label='minima_interpolation_y')
            # plt.plot(interpolation_x, interpolation_y, label='interpolation_y')
            # plt.legend()
            # plt.show()
            # plt.close()
            # input("A")
        return interpolation_x, interpolation_y



    def interpolate_extrema_subsection(self, x, y):
        """
        分段获取每段的极大值极小值，如果没有，则间隔取原始数据
        所有段的极大值和极小值插值，得到中间的线
        :param x:
        :param y:
        :return:
        """
        # if self.get_cal_smothness(y) > 0.05:
        #     y = savgol_filter(y, window_length=min(len(y), 10), polyorder=1)

        if len(y) > 0:
            # y = savgol_filter(y, window_length=min(len(y), 10), polyorder=1, mode='interp')
            # SG滤波会影响数据首部的形态
            # y = MVFilter(20, outlier_k=3).predict(y)

            # 两侧填充后再滤波 240223
            pad_width = 20
            padding_y = edge_padding(y, pad_width)
            padding_y = MVFilter(20, outlier_k=3).predict(padding_y)

            y = padding_y[pad_width:-pad_width]
            # 使用三次样条插值
            # f_cubic = interp1d(range(len(y)), y, kind='cubic')
            # y = f_cubic(range(len(y)))


        if self.segment_num is None:
            self.segment_num = len(y) + 1
        subsection_num = min(int(len(y) / self.segment_num) + 1, len(y))
        x_split = np.array_split(x, subsection_num)
        y_split = np.array_split(y, subsection_num)
        maxima_idx = []
        minima_idx = []
        need_add = 0
        for index, part_y in enumerate(y_split):
            part_maxima_idx = argrelextrema(part_y, np.greater)[0] + need_add
            part_minima_idx = argrelextrema(part_y, np.less)[0] + need_add
            limit_min_num = 0
            if len(part_maxima_idx) <= limit_min_num:
                part_maxima_idx = x_split[index][::max(int(self.segment_num / 10), 3)]
            if len(part_minima_idx) <= limit_min_num:
                part_minima_idx = x_split[index][::max(int(self.segment_num / 10), 3)]

            need_add = need_add + len(part_y)
            # 计算极大值和极小值的索引
            maxima_idx.extend(part_maxima_idx)
            minima_idx.extend(part_minima_idx)
        maxima_idx = np.array(maxima_idx)
        minima_idx = np.array(minima_idx)

        # 提取极大值和极小值的坐标
        maxima_x = x[maxima_idx]
        maxima_y = y[maxima_idx]
        minima_x = x[minima_idx]
        minima_y = y[minima_idx]

        # 231204
        interpolation_x, interpolation_y = self.data_interpolation_greater_less(x, y, maxima_x, maxima_y, minima_x,
                                                                                minima_y)

        return interpolation_x, interpolation_y

    def fit(self, signal: np.ndarray, **kwargs):
        """
        短时傅里叶变换
        :return:
        """
        signal = np.array(signal)
        filtered_signal_x, filtered_signal = self.interpolate_extrema_subsection(np.array(range(len(signal))), signal)

        return filtered_signal

    def predict(self, x):
        return self.fit(x)


func_class_dict = {'wave': WaveFilter,
                   'emd': EmdFilter,
                   'eemd': EEmdFilter,
                   'stft': STFTFilter,
                   'extremum': ExtremumFilter,
                   'mv':MVFilter}


class FilterSmooth:
    def __new__(cls, model_type, *args, **kwargs):
        return func_class_dict[model_type](*args, **kwargs)
