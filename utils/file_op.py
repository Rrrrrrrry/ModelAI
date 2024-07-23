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

def detect_encoding(path):
    """
    check encoding
    :return:
    """
    with open(path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    return encoding


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
    获取csv格式的测试曲线数据
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


class TxtFileRead:
    """
    获取txt格式的测试曲线数据
    """
    @staticmethod
    def detect_separator(file_path, num_lines=5, possible_separators=None):
        """
        自动检测txt中的分隔符
        :param file_path:
        :param num_lines:
        :param possible_separators:常见分隔符
        :return:
        """
        this_file_encoding = detect_encoding(file_path)
        if this_file_encoding not in ['utf8', 'utf-8']:
            this_file_encoding = 'gbk'
        if possible_separators is None:
            possible_separators = [',', '\t', ';', '|']
        with open(file_path, 'r', encoding=this_file_encoding) as file:
            lines = [file.readline() for _ in range(num_lines)]

        best_guess = ' '
        max_count = 0
        for sep in possible_separators:
            counts = [line.count(sep) for line in lines[0:1]]
            if counts[0] > max_count:
                best_guess = sep
                max_count = counts[0]
        return best_guess, this_file_encoding

    @staticmethod
    def clean_dataframe(df):
        """
        清理首尾空格
        :param df:
        :return:
        """
        df.columns = df.columns.str.strip()
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df

    @staticmethod
    def read_file(file_path):
        """
        解析多种分隔符的txt文档
        :param file_path:字符串，包含文件路径和文件名的完整路径。
        "./datasets/test.txt"
        :return:
        读取txt格式的数据，如果数据不存在，则返回pd.DataFrame()
        """
        if not os.path.exists(file_path):
            return pd.DataFrame()
        separator, this_file_encoding = TxtFileRead.detect_separator(file_path)
        if separator == ' ':
            df = pd.read_csv(file_path, delim_whitespace=True, index_col=False, encoding=this_file_encoding)
        else:
            df = pd.read_csv(file_path, delimiter=separator, index_col=False, encoding=this_file_encoding)
        df = TxtFileRead.clean_dataframe(df)
        return df



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

# if __name__ == '__main__':
#     from pathlib import Path
#     import sys
#     root_path = str(Path(sys.path[0]).resolve().parents[0])
#     data_path = os.path.join(root_path, 'datasets', 'txt', 'test.txt')
#     data = TxtFileRead.read_file(data_path)
#     print(data)