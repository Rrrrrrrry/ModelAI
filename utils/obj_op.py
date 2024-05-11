import inspect


def get_valid_parameters(obj):
    """
    获取obj的有效参数
    :param obj:
    :return:
    """
    valid_params = inspect.signature(obj).parameters.keys()
    return valid_params

def filter_param(param_dict, data_object):
    """
    过滤参数，保证每个参数都存在于类object的构造函数列表内
    :param data_object:
    :return:
    """
    object_params = inspect.signature(data_object).parameters
    param_names = list(object_params.keys())
    filtered_params = {k: v for k, v in param_dict.items() if k in param_names}
    return filtered_params