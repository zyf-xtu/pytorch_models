# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 18:03
# @Author  : zyf
# @File    : Darknet53.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    复现YOLOV3论文中的网络结构Darknet53
    该网络结构引入了残差模块，但具体的残差结构又跟ResNet的残差结构不同，具体的残差结构请看残差模块的设计。
    卷积类、残差类、主干网络
    输入：3x256x256
    共包含53层网络
    其结构如下：
    常规卷积(1)->下采样卷积(1)->残差模块(1x2)->下采样卷积(1)->残差模块(2x2)
                                                           /
    ------------------------------------------------------
     \
       ->下采样卷积(1)->残差模块(8x2)->下采样卷积(1)->残差模块(8x2)->下采样卷积(1)->残差模块(4x2)->自定义池化avg_pool->全连接fc
    1+1+2+1+4+1+16+1+16+1+8+fc = 53
    具体结构变化见代码详细注释
'''


# 定义卷积类
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        """
        构建卷积类，包含一个卷积、池化、归一化
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param kwargs: 自定义参数
        """
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


# 定义残差类
class Residual(nn.Module):
    def __init__(self, in_ch):
        """
        定义残差模块，包含两个卷积，第一个是1x1的卷积，进行降维，然后经过3x3的卷积升维。
        :param in_ch:输入的通道数
        """
        super(Residual, self).__init__()
        # 对输入的通道数进行2倍降维
        ch = in_ch // 2
        self.conv = nn.Sequential(
            ConvLayer(in_ch=in_ch, out_ch=ch, kernel_size=1),  # 1x1的卷积进行降维
            ConvLayer(in_ch=ch, out_ch=in_ch, kernel_size=3, padding=1)  # 3x3的卷积升维
        )

    def forward(self, x):
        out = x + self.conv(x)  # 残差相加，既保留原来的信息，又融入了新提取的特征
        return out


# 定义Darknet53的网络结构
class Darknet53(nn.Module):
    def __init__(self, num_classes=1000):
        """
        复现网络结构，这里用作分类
        :param num_classes: 分类数
        """
        super(Darknet53, self).__init__()
        self.num_classes = num_classes
        # 常规卷积：3x256x256 ->32x256x256
        self.conv1 = ConvLayer(in_ch=3, out_ch=32, kernel_size=3, padding=1)
        # 一个下采样的卷积：32x256x256 -> 64x128x128
        self.conv2 = ConvLayer(in_ch=32, out_ch=64, kernel_size=3, stride=2, padding=1)
        # 第一个残差结构，包含一个残差块
        self.residual1 = Residual(in_ch=64)
        # 一个下采样的卷积:64x128x128 -> 128x64x64
        self.downsample_conv1 = ConvLayer(in_ch=64, out_ch=128, kernel_size=3, stride=2, padding=1)
        # 第二个残差结构，包含两个残差块
        self.residual2 = nn.Sequential(
            Residual(in_ch=128),
            Residual(in_ch=128)
        )
        # 一个下采样的卷积:128x64x64 -> 256x32x32
        self.downsample_conv2 = ConvLayer(in_ch=128, out_ch=256, kernel_size=3, stride=2, padding=1)
        # 第三个残差结构，包含8个残差块
        self.residual3 = nn.Sequential(
            *[Residual(in_ch=256) for _ in range(8)]
        )
        # 一个下采样的卷积：256x32x32 -> 512x16x16
        self.downsample_conv3 = ConvLayer(in_ch=256, out_ch=512, kernel_size=3, stride=2, padding=1)
        # 第四个残差结构，包含8个残差块
        self.residual4 = nn.Sequential(
            *[Residual(in_ch=512) for _ in range(8)]
        )
        # 一个下采样的卷积：512x16x16 -> 1024x8x8
        self.downsample_conv4 = ConvLayer(in_ch=512, out_ch=1024, kernel_size=3, stride=2, padding=1)
        # 第五个残差结构，包含4个残差块
        self.residual5 = nn.Sequential(
            *[Residual(in_ch=1024) for _ in range(8)]
        )
        # 自定义池化，固定输出尺寸
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(1024, self.num_classes)

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual1(x)
        x = self.downsample_conv1(x)
        x = self.residual2(x)
        x = self.downsample_conv2(x)
        x = self.residual3(x)
        x = self.downsample_conv3(x)
        x = self.residual4(x)
        x = self.downsample_conv4(x)
        x = self.residual5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


x = torch.rand((1, 3, 256, 256))
model = Darknet53()
print(model)
out = model(x)
print(out)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
summary(model, (3, 256, 256))
