import numpy as np


def sigma_del_outliers(data, k=3):
    """
    sigma 异常值检测方法
    :param data:
    :return:
    """
    data = np.array(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    down_threshold = data_mean - k * data_std
    up_threshold = data_mean + k * data_std
    del_outliers_data = [v for i, v in enumerate(data) if (v >= down_threshold or v <= up_threshold)]
    return del_outliers_data

def edge_padding(data, pad_width):
    """
    对数据进行边缘填充
    """
    # 复制数据
    mirrored_data = np.pad(data, pad_width, mode='edge')
    return mirrored_data