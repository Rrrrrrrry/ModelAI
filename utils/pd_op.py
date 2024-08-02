import re
from io import StringIO

import numpy as np
import pandas as pd


def from_pd_get_confusion_matrix(pd_unit):
    """
    从pd的某一单元格内，解析混淆矩阵的str为np
    :param pd_unit:
    :return:
    """
    # 解析字符串
    confusion_matrix_str = pd_unit[1:-1]
    parsed_matrix = re.findall(r'\d+', confusion_matrix_str)
    parsed_matrix = list(map(int, parsed_matrix))
    rows = confusion_matrix_str.count('[')
    cols = len(parsed_matrix) // rows
    # # 将解析后的列表转换为二维数组
    confusion_matrix = np.array(parsed_matrix).reshape(rows, cols)
    print(f"confusion_matrix{confusion_matrix}")
    print(f"confusion_matrix{type(confusion_matrix)}")
    return confusion_matrix


def from_pd_get_acc_reports(pd_unit):
    """
    从pd的某一单元格内，解析准确率的report的str为pd(解析的不完整，待完善)
    :param pd_unit:
    :return:
    """
    lines = pd_unit.strip().split('\n')
    # 提取表头和数据
    header = " ".join(lines[0].strip().split()).split()
    data = [line.split() for line in lines[1:] if line.strip()]
    # 创建DataFrame
    report_data = pd.DataFrame(data)
    return report_data

# if __name__ == '__main__':
#     data = pd.read_excel(r'D:\python_program\ModelAI\datasets\excel/all_accuracies_train.xlsx')
#     for i, row in data.iterrows():
#         # from_pd_get_confusion_matrix(row['混淆矩阵'])
#         from_pd_get_acc_reports(row['report'])
#         input('s')


