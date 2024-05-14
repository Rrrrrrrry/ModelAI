import math
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

currentdir = os.path.dirname(os.path.abspath(sys.path[0]))
sys.path.append(currentdir)
from algorithms.machine_learning.feature_extraction.filter_smooth import *
if __name__ == '__main__':
    this_well_crv_data = np.random.random(100)

    # 曲线平滑
    mode_list = ['wave', 'emd', 'eemd', 'stft', 'mv', 'sg']
    # mode_list = ['sg', 'stft', 'mv']
    num_cols = 3  # 每行显示的最大列数
    num_rows = math.ceil(len(mode_list) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    # 确保axes是二维数组
    if num_rows == 1:
        axes = np.reshape(axes, (1, num_cols))
    elif num_cols == 1:
        axes = np.reshape(axes, (num_rows, 1))
    # 作图可视化，标准段和当前井的地层段的直方图叠合图
    j = 0
    print(f"num_rows{num_rows}")
    for i, mode in enumerate(mode_list):
        row = i // num_cols  # 计算当前方法应该放置的行数
        col = i % num_cols  # 计算当前方法应该放置的列数

        plt.sca(axes[row, col])

        # plt.figure()
        # mode = 'stft'
        filter_data_wave = FilterSmooth(mode, **{'segment_num': 20, 'window_size': 20,
                                                 **{'polyorder': 2, mode:'nearest', 'o':22}}).fit(this_well_crv_data)[
                           0:len(this_well_crv_data)]
        plt.plot(this_well_crv_data, color='g')
        plt.plot(filter_data_wave, color='r')
        plt.title(f"{mode}")
    plt.show()

    # # mode = 'mv'
    # mode = 'extremum'
    # filter_data_wave = FilterSmooth(mode, **{'segment_num':20}).fit(data)[0:len(data)]
    # print(f"data{data}")
    # print(f"filter_data_wave{filter_data_wave}")
