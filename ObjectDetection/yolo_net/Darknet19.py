#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/4 20:03
# @Author  : zyf
# @File    : Darknet19.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    复现YOLOv2论文中的darknet19网络结构
    YOLOv2去掉了YOLOv1中的全连接层，使用anchor boxes预测边界框。
    使用416x416的输入，模型下采样的总步长为32，最后得到13x13的特征图。
    Darknet-19,包括19个卷积层和5个max pooling层，主要采用3x3和1x1卷积
    1x1卷积可以压缩特征图通道数降低模型计算量和参数，每个卷积层后使用BN层以加快模型收敛同时防止过拟合。
    最终采用global avg pool 做预测。
    Darknet-19 结构类似VGG结构
'''


# 构建卷积层
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        """
        构建卷积层，包含一个卷积、BN、ReLu
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param kwargs:
        """
        super(ConvLayer, self).__init__()
        # 卷积
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, **kwargs)
        # 批量归一化
        self.bn = nn.BatchNorm2d(out_ch)
        # 激活函数
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 构建darknet-19网络
class DarkNet19(nn.Module):
    def __init__(self, num_classes=1000):
        """
        复现网络结构，这里用作分类
        :param num_classes: 分类数
        """
        super(DarkNet19, self).__init__()
        self.num_classes = num_classes
        # 第1,2层卷积
        self.conv1 = nn.Sequential(
            # 第一个卷积:3*448*448 -> 32*224*224
            ConvLayer(in_ch=3, out_ch=32, kernel_size=3, stride=2, padding=1),
            # 最大池化：32*224*224 -> 32*112*112
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第二个卷积:32*112*112 -> 64*112*112
            ConvLayer(in_ch=32, out_ch=64, kernel_size=3, padding=1),
            # 最大池化：64*112*112 -> 64*56*56
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第3,4，5层卷积
        self.conv2 = nn.Sequential(
            # 第三个卷积：64*56*56 -> 128*56*56
            ConvLayer(in_ch=64, out_ch=128, kernel_size=3, padding=1),
            # 第四个卷积：128*56*56 -> 64*56*56
            ConvLayer(in_ch=128, out_ch=64, kernel_size=1),
            # 第五个卷积：64*56*56 -> 128*56*56
            ConvLayer(in_ch=64, out_ch=128, kernel_size=3, padding=1),
            # 最大池化：128*56*56 -> 128*28*28
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第6,7，8层卷积
        self.conv3 = nn.Sequential(
            # 第六个卷积：128*28*28 -> 256*28*28
            ConvLayer(in_ch=128, out_ch=256, kernel_size=3, padding=1),
            # 第七个卷积：256*28*28 -> 128*28*28
            ConvLayer(in_ch=256, out_ch=128, kernel_size=1),
            # 第八个卷积：128*28*28 -> 256*28*28
            ConvLayer(in_ch=128, out_ch=256, kernel_size=3, padding=1),
            # 最大池化：256*28*28 -> 256*14*14
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第9,10,11,12,13层卷积
        self.conv4 = nn.Sequential(
            # 第9个卷积：256*14*14 -> 512*14*14
            ConvLayer(in_ch=256, out_ch=512, kernel_size=3, padding=1),
            # 第10个卷积：512*14*14 -> 256*14*14
            ConvLayer(in_ch=512, out_ch=256, kernel_size=1),
            # 第11个卷积：256*14*14 -> 512*14*14
            ConvLayer(in_ch=256, out_ch=512, kernel_size=3, padding=1),
            # 第12个卷积：512*14*14 -> 256*14*14
            ConvLayer(in_ch=512, out_ch=256, kernel_size=1),
            # 第13个卷积：256*14*14 -> 512*14*14
            ConvLayer(in_ch=256, out_ch=512, kernel_size=3, padding=1),
            # 最大池化：512*14*14 -> 512*7*7
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第14,15,16,17,18层卷积
        self.conv5 = nn.Sequential(
            # 第14个卷积：512*7*7 -> 1024*7*7
            ConvLayer(in_ch=512, out_ch=1024, kernel_size=3, padding=1),
            # 第15个卷积：1024*7*7 -> 512*7*7
            ConvLayer(in_ch=1024, out_ch=512, kernel_size=1),
            # 第16个卷积：512*7*7 -> 1024*7*7
            ConvLayer(in_ch=512, out_ch=1024, kernel_size=3, padding=1),
            # 第17个卷积：1024*7*7 -> 512*7*7
            ConvLayer(in_ch=1024, out_ch=512, kernel_size=1),
            # 第18个卷积：512*7*7 -> 1024*7*7
            ConvLayer(in_ch=512, out_ch=1024, kernel_size=3, padding=1)
        )
        # 第19层卷积：1024*7*7 -> 1000*7*7
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1)
        # 自定义池化，固定输出尺寸：1000*7*7 -> 1000*1*1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(1000 * 1 * 1, 1000)

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


x = torch.rand((1, 3, 448, 448))
model = DarkNet19()
print(model)
out = model(x)
print(out)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
summary(model, (3, 448, 448))
