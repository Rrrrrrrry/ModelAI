# 数据相关

Urban Drone Dataset（UDD）数据集是北京大学图形与交互实验室采集并标注的，面向航拍场景理解、重建的数据集。

![image-20240424165506708](C:\Users\RenYu\AppData\Roaming\Typora\typora-user-images\image-20240424165506708.png)



# 代码相关

最终采用的模型是‘基于mobileNet_v2的unet模型’和‘基于mobileNet_v2的pspnet模型’，具体原因如下：

1、unet支持不同大小的图像输入，训练的时候不需要全部resize，但是需要变为32的倍数，且batch为1，但是图像太大会导致速度太慢了

修改位置-main.py：

```
train_loader = get_dataloader(dataloader, batch_size=8)


def resize_with_label(image, label, new_size, interpolation=cv2.INTER_NEAREST, multiple_num=32):
    # 需要调整到32的倍数
    # # 计算宽度和高度的32的倍数
    # width, height, _ = image.shape
    # new_width = ((width - 1) // multiple_num + 1) * multiple_num
    # new_height = ((height - 1) // multiple_num + 1) * multiple_num
    # new_size = [new_width, new_height]
```

2、spp的核心是为了支持不同图像大小的输入，它允许网络在不同尺度上对输入图像进行池化操作，并将所有尺度下的特征串联在一起，从而提高了网络对于不同尺度物体的检测和分割能力。通常和spp和全连接层是连接在一起的

3、unet_spp脚本有问题，暂时无法使用

初步认为unet-spp无法结合，原因如下：

```
UNet（2015年提出）:
UNet是一种经典的全卷积网络-允许不同大小的图像输入，专门用于图像分割任务。其结构由编码器和解码器组成，编码器负责逐渐降低输入图像的空间分辨率并提取高级特征，而解码器则负责将这些特征映射回输入图像的空间尺寸，并生成语义分割的结果。UNet以U形状的结构而得名，其中跨层连接有助于将低级和高级特征进行融合，从而提高了分割性能。

SPP（2014年提出）-spatial pyramid pooling layer(空间金字塔池化层):
SPP是一种空间金字塔池化技术，提出的目的是当初的神经网络不支持任意大小的输入。spp通过将输入特征图划分为不同尺度的金字塔区域，并对每个区域应用池化操作，最后将所有池化结果串联起来，使网络支持任意大小的输入。spp层通常加在全连接层之前。

unet-spp无法结合（以初步调查结果判定的）
1、unet是全卷积网络，没有全连接层，本身就支持任意图像的输入
2、现在有较新的类似于金字塔池化的模型PSPNet

PSPNet(2016年)-Pyramid Scene Parseing Network,采用金字塔池化模块搭建的场景分析网络:
PSPNet是一种用于语义分割任务的深度学习架构，旨在通过金字塔池化模块（PSP module）来捕捉不同尺度的上下文信息，从而改善语义分割的性能。
```

4、模型的内部均采用了mobilenet_v2，原因是轻量、速度快

# 结果相关

模型一：unet_240423，采用unet-mobilenet_v2本地训练了100代，测试结果

```
pixel_accuracy, iou:(0.815849922558309, 0.6889751038322104)
```

模型二：采用unet-mobilenet_v2服务器训练了200代，测试结果

```
pixel_accuracy, iou:(0.7893198797376093, 0.6519640213193061)
```

模型三：采用pspnet-mobilenet_v2服务器训练了200代，测试结果

```
pixel_accuracy, iou:(0.7549312135568513, 0.6063369524454161)
```



# 涉及到smp库源码修改(可能不需要修改源码，但是暂时没找到其他方法)

## 使用模型参数的保存路径

由于smp内部需要下载url指定路径，没有外网的情况下无法下载，此时需要加载下载好的.pth文件，正常默认的路径为PyTorch Hub 默认缓存目录路径：/root/.cache/torch/hub/checkpoints

```
import torch
# 获取 PyTorch Hub 的默认缓存目录
hub_dir = torch.hub.get_dir()
model_dir = os.path.join(hub_dir, 'checkpoints')
```

如果不想改源码，就将模型.pth放到PyTorch Hub 默认缓存目录路径，否则需要修改源码中的路径

## 路径修改

为了方便后期部署调试，将hub.py下的模型保存路径改为当前项目路径下的models_use路径

```
# 240424修改模型保存的路径
model_dir = os.path.join(sys.path[0], 'models_use')
```

## hub.py位置检索方式

- 第一种：直接检索hub.py路径，修改model_dir

![image-20240424200535844](C:\Users\RenYu\AppData\Roaming\Typora\typora-user-images\image-20240424200535844.png)

![image-20240424200645054](C:\Users\RenYu\AppData\Roaming\Typora\typora-user-images\image-20240424200645054.png)



- 第二种：按文件方式跳转检索

找到smp.PSPNet

![image-20240424200244263](C:\Users\RenYu\AppData\Roaming\Typora\typora-user-images\image-20240424200244263.png)

跳转到model.py，找到get_encoder

![image-20240424200316897](C:\Users\RenYu\AppData\Roaming\Typora\typora-user-images\image-20240424200316897.png)

跳转到_init_.py下，找到load_url

![image-20240424200417014](C:\Users\RenYu\AppData\Roaming\Typora\typora-user-images\image-20240424200417014.png)

跳转到hub.py下找到model_dir的修改位置

![image-20240424200504672](C:\Users\RenYu\AppData\Roaming\Typora\typora-user-images\image-20240424200504672.png)



