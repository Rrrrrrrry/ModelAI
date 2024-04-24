import inspect


def get_valid_parameters(obj):
    """
    获取obj的有效参数
    :param obj:
    :return:
    """
    valid_params = inspect.signature(obj).parameters.keys()
    return valid_params