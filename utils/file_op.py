import glob
import os
import numpy as np
import chardet
import pandas as pd
"""
文件操作工具file_op:
用于文件读写、目录管理和文件格式转换的函数。
用于管理数据集文件、保存和加载模型参数等。
"""
class ExcelFileRead:
    """
    获取excel格式的数据
    """
    @staticmethod
    def read_file(file_path):
        """
        :param file_path:字符串，包含文件路径和文件名的完整路径。
        "./datasets/test.csv"
        :return:
        读取excel格式的测井曲线数据，如果数据不存在，则返回pd.DataFrame()
        """
        if not os.path.exists(file_path):
            return pd.DataFrame()
        curves_datas = pd.read_excel(file_path)
        return curves_datas


class CsvFileRead:
    """
    获取excel格式的测试曲线数据
    """
    @staticmethod
    def read_file(file_path):
        """
        :param file_path:字符串，包含文件路径和文件名的完整路径。
        "./datasets/test.csv"
        :return:
        读取csv格式的数据，如果数据不存在，则返回pd.DataFrame()
        """
        if not os.path.exists(file_path):
            return pd.DataFrame()
        f = open(file_path, 'rb')
        f_charinfo = chardet.detect(f.read())['encoding']
        curves_datas = pd.read_csv(open(file_path, encoding=f_charinfo))
        return curves_datas



def save_npy(data, file_path):
    """
    np数据保存为npy文件
    :param data:
    :param file_path:
    :return:
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, np.array(data))


def load_npy(file_path):
    """
    npy文件的加载
    :param file_path:
    :return:
    """
    if os.path.exists(file_path):
        data = np.load(file_path)
    else:
        data = []
    return data

def glob_select_file(root_path, select_file_name):
    """
    从根目录下遍历搜索所有名为select_file_name的文件
    :param root_path:根路径
    :param select_file_name:待搜索的文件名
    :return:所有文件路径
    """
    files = glob.glob(os.path.join(root_path, '**', select_file_name), recursive=True)
    return files
