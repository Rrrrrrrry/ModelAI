import itertools
import logging
import os
import sys
import traceback

sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))
from utils.config_utils import *
if __name__=="__main__":
    choose_model_name_options = ['rf', 'mlp', 'vot']
    filter_label_list_options = [['A22'], ['A2'], ['E3']]
    train_well_file_name_options = [['214_out.csv', '217_out.csv', 'N201_20240301.csv', 'N209_20240301.csv'],
                                    ['214_out.csv', '217_out.csv', 'N201_20240301.csv', 'N211s_20240301.csv'],
                                    ]



    float_test = [2]

    config_file = os.path.join(os.path.dirname(sys.path[0]), 'config', 'config_test.py')

    # 遍历所有组合
    for choose_model_name, filter_label_list, train_well_file_name, float_test_data in itertools.product(
            choose_model_name_options,
            filter_label_list_options,
            train_well_file_name_options,
            float_test):
        try:
            print(f"choose_model_name32{choose_model_name}")
            print(f"filter_label_list{filter_label_list}")
            print(f"train_well_file_name_combination{train_well_file_name}")
            # 修改配置文件
            # modify_config_file(config_file, choose_model_name, filter_label_list, train_well_file_name_combination)
            modify_config_file(config_file, {"choose_model_name": choose_model_name,
                                             "filter_label_list": filter_label_list,
                                             "train_well_file_name": train_well_file_name,
                                             "data_test": float_test_data})
            # 调脚本
            # call_lith_py()
        except Exception as e:
            logging.exception("An error occurred: %s", str(e))
            traceback.print_exc()
            # f = open(os.path.join(log_save_path, f'{log_txt_name}_error.txt'), 'a')
            # f.write(f"choose_model_name->{choose_model_name}\n")  # 12.5
            # f.write(f"filter_label_list->{filter_label_list}\n")  # 12.5
            # f.write(f"train_well_file_name_combination->{train_well_file_name_combination}\n")
            # traceback.print_exc(file=f)
            # f.flush()
            # f.close()
            sys.exit(1)  # 终止程序

