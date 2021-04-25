#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/15 17:45
# @Author  : zyf
# @File    : vgg.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

"""
    实现vgg16和vgg19,用来作为fcn的骨干网络
    vgg16的参数表：[2,2,3,3,3]
    vgg16的参数表：[2,2,4,4,4]
"""


# 制作卷积
def make_layers(depth, in_channel, out_channel):
    layers = []
    for i in range(depth):
        conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        bn = nn.BatchNorm2d(out_channel)
        relu = nn.ReLU()
        # layers +=[conv,bn,relu]
        layers.extend([conv, bn, relu])
        in_channel = out_channel
    # print(layers)
    # print(*layers)
    return nn.Sequential(*layers)


class vgg(nn.Module):
    def __init__(self, num_classes=1000, layers=16):
        """
        实现vgg16和19
        :param num_classes: 分类数
        :param layers: 层数
        """
        super(vgg, self).__init__()
        self.num_classes = num_classes
        self.layers = layers
        supported_layers = [16, 19]  # 支持的层数只有16和19
        assert layers in supported_layers  # 断言
        if layers == 16:
            depth = [2, 2, 3, 3, 3]  # 16= 2+2+3+3+3
        elif layers == 19:
            depth = [2, 2, 4, 4, 4]  # 19 = 2+2+4+4+4
        # 输入通道数
        in_channels = [3, 64, 128, 256, 512]
        # 输出通道数
        out_channels = [64, 128, 256, 512, 512]
        # 第一个卷积部分
        self.layer1 = nn.Sequential(make_layers(depth[0], in_channels[0], out_channels[0]))
        # 第一个池化部分
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积部分
        self.layer2 = nn.Sequential(make_layers(depth[1], in_channels[1], out_channels[1]))
        # 第二个池化部分
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三个卷积部分
        self.layer3 = nn.Sequential(make_layers(depth[2], in_channels[2], out_channels[2]))
        # 第三个池化部分
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第四个卷积部分
        self.layer4 = nn.Sequential(make_layers(depth[3], in_channels[3], out_channels[3]))
        # 第四个池化部分
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第五个卷积部分
        self.layer5 = nn.Sequential(make_layers(depth[4], in_channels[4], out_channels[4]))
        # 第五个池化部分
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 自适应池化Adaptive Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        # 全连接层
        self.classifer = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.pool4(x)
        x = self.layer5(x)
        x = self.pool5(x)


# data = torch.rand((1,3,224,224))
# print(data)
# net = vgg()
# print(net)
# out = net(data)
