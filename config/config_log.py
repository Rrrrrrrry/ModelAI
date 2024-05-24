import logging
import os
import sys


def config_log(this_file_path=os.path.join(sys.path[0], 'config_log.log'), basicConfig_level=logging.INFO, console_level=logging.INFO):
    """
    配置log日志
    :param this_file_path: log文件保存路径
    :param basicConfig_level:日志文件的level[debug, info, warning, error, critical]
    :param console_level:终端处理器的level[debug, info, warning, error, critical]
    :return:
    """
    this_file_path_dir = os.path.dirname(os.path.abspath(this_file_path))
    os.makedirs(this_file_path_dir, exist_ok=True)

    # 配置日志记录器
    logging.basicConfig(filename=this_file_path, level=basicConfig_level, format='%(asctime)s - %(levelname)s - %(message)s')
    # 创建日志记录器
    logger = logging.getLogger()

    # 创建终端处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 将终端处理器添加到日志记录器
    logger.addHandler(console_handler)
    return logger

