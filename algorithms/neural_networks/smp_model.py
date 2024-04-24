"""
使用图像分割的库函数segmentation_models_pytorch定义模型
"""
import os.path
import sys
import segmentation_models_pytorch as smp

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0])))
sys.path.append(current_dir)
from utils.obj_op import get_valid_parameters

def smp_unet(*args, **kwargs):
    # 默认参数值
    default_kwargs = {
        'encoder_name': 'mobilenet_v2',
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 6,
        'activation': None,
        'encoder_depth': 5,
        'decoder_channels': [256, 128, 64, 32, 16]
    }
    # 更新默认参数值
    default_kwargs.update(kwargs)
    unet_params = get_valid_parameters(smp.Unet)
    # 筛选出不在smp.Unet构造函数内的参数
    default_kwargs = {k: v for k, v in default_kwargs.items() if k in unet_params}
    # 使用更新后的参数值创建模型
    model = smp.Unet(*args, **default_kwargs)
    return model


def smp_pspnet(*args, **kwargs):

    # 默认参数值
    default_kwargs = {
        'encoder_name': 'mobilenet_v2',
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 6,
    }

    # 更新默认参数值
    default_kwargs.update(kwargs)
    unet_params = get_valid_parameters(smp.PSPNet)
    default_kwargs = {k: v for k, v in default_kwargs.items() if k in unet_params}
    # 使用更新后的参数值创建模型
    model = smp.PSPNet(*args, **default_kwargs)

    return model


