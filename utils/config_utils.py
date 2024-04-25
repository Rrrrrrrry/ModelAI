"""
不完善
"""
import configparser
import logging
import os
import sys
from utils.utils import is_evaluable

config = configparser.ConfigParser()

def set_config():
    # 添加配置项和值
    config['section_name'] = {
        'option1': [1, 3, 4],
        'option2': 'value2',
        'option3': False
    }
    # config.set('section_name', 'option1', 'new_value')

    # 保存配置文件
    with open('config.ini', 'w') as config_file:
        config.write(config_file)


def get_config():
    # 读取配置文件
    config.read('config.ini')
    # 获取所有的节
    sections = config.sections()
    print("sections", sections)
    # 遍历每个节并获取其中的配置项
    for section in sections:
        options = config.options(section)
        for option in options:
            value = config.get(section, option)
            if is_evaluable(value):
                value = eval(value)
            print(f"{section} -> {option} = {value}")

    # # 获取配置项的值
    # value = config.get('section_name', 'option1')
    # value = eval(value)
    # print("value", value, type(value))
# set_config()
# get_config()

def set_yaml_config():
    import yaml
    # 创建配置数据
    config_data = {
        'name': 'John Smith',
        'age': 30,
        'address': {
            'street': '123 Main St',
            'city': 'Anytown',
            'country': 'USA'
        }
    }
    # 将配置数据写入配置文件
    with open('config.yaml', 'w') as file:
        yaml.dump(config_data, file)


def get_yaml_config():
    import yaml
    # 读取配置文件
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    for section, values in config.items():
        print(f"section: {section}, {values}")

    # value = config['section_name']['option_name']
    # # 修改配置项的值
    # config['section_name']['option_name'] = 'new_value'
    #
    # # 保存修改后的配置文件
    # with open('config.yaml', 'w') as config_file:
    #     yaml.safe_dump(config, config_file)


def config_log(this_file_path=os.path.join(sys.path[0], 'filter.log')):
    """
    配置log
    debug, info, warning, error, critical
    :return:
    """
    this_file_path_dir = os.path.dirname(os.path.abspath(this_file_path))
    os.makedirs(this_file_path_dir, exist_ok=True)

    # 配置日志记录器
    logging.basicConfig(filename=this_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # 创建日志记录器
    logger = logging.getLogger()

    # 创建终端处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 将终端处理器添加到日志记录器
    logger.addHandler(console_handler)
    return logger
