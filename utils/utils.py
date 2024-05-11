import hashlib
import ast
from pypinyin import pinyin, Style

def chinese_to_upper_case(string):
    """
    获取中文对应的大写字母，英文不变
    :param string:
    :return:
    """
    # 使用 pypinyin 库将中文字符串转换为拼音
    pinyin_result = pinyin(string, style=Style.FIRST_LETTER)

    # 提取拼音首字母，并转换为大写
    first_letters = [x[0].upper() for x in pinyin_result if x[0].isalpha()]

    # 将提取的拼音首字母合并成一个字符串
    return ''.join(first_letters)

def baz(*args, **kwargs):
    """
    baz(1, 2, 3, name='Alice', age=25)
    :param args: 接受任意数量的位置参数（非关键字参数），它会将传递给函数的参数打包成一个元组（tuple）
    :param kwargs:接受任意数量的关键字参数，它会将传递给函数的关键字参数打包成一个字典（dictionary）
    :return:
    """
    for arg in args:
        print(arg)
    for key, value in kwargs.items():
        print(key, value)


def is_evaluable(data):
    """
    判断数据是否可以被eval()
    :param data:
    :return:
    """
    try:
        ast.literal_eval(data)
        return True
    except (SyntaxError, ValueError):
        return False


def calculate_file_hash(filename):
    # 创建哈希对象
    hasher = hashlib.md5()

    # 打开文件并逐块计算哈希值
    with open(filename, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            hasher.update(chunk)

    # 返回计算得到的哈希值
    return hasher.hexdigest()


def has_file_changed(filename, previous_hash=None):
    """
    判断文件是否被修改过
    :param filename:
    :param previous_hash:
    :return:
    """
    if previous_hash is None:
        # 如果没有提供之前的哈希值，则读取之前保存的哈希值（假设保存在文件中）
        try:
            with open('previous_hash.txt', 'r') as file:
                previous_hash = file.read()
        except FileNotFoundError:
            # 如果文件不存在，将哈希值设为 None
            previous_hash = None

    # 计算当前文件的哈希值
    current_hash = calculate_file_hash(filename)

    # 比较当前哈希值和之前的哈希值
    if current_hash != previous_hash:
        print("文件已发生变化")
        # 更新保存的哈希值为当前哈希值
        with open('previous_hash.txt', 'w') as file:
            file.write(current_hash)
    else:
        print("文件未发生变化")
