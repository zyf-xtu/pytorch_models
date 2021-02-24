#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 15:02
# @Author  : zyf
# @File    : VGG.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    已经单独实现了VGG16、VGG19等VGG网络，但是由于是纯代码堆叠，比较臃肿，这里仿照VGG官方重写一个VGG网络，
    通过参数、模块化实现不同层的网络结构。
    代码结构：
        主体结构:VGG
        卷积函数:make_layers
'''

# 所实现的vgg网络结构,*_bn 表示结构中带有BatchNorm2d归一化
_all_ = ['VGG11', 'VGG13', 'VGG16', 'VGG19', 'VGG11_bn', 'VGG13_bn', 'VGG16_bn', 'VGG19_bn']


# 主体结构
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        # 分类数
        self.num_classes = num_classes
        # 卷积提取特征部分，这部分纯手写在VGG_16.py和VGG_19.py中已经实现，详细请参考
        self.features = features
        # 自定义池化，目的固定输出size
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        # 全连接部分
        self.classifier = nn.Sequential(
            # 第一个全连接层fc1,输入：batch_size*512*7*7  输出：4096
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 第二个全连接层fc2,输入：4096  输出：4096
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 第三个全连接层fc3,输入：4096  输出：nums 即分类数
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    # 前向传播
    def forward(self, x):
        # 卷积部分
        x = self.features(x)
        # 自定义池化
        x = self.avg_pool(x)
        # 维度展开
        x = torch.flatten(x, 1)
        # 全连接部分
        x = self.classifier(x)
        return x


# 定义卷积结构，因为卷积都是3*3的卷积核，核心在于不同层的参数不一样。
def make_layers(cfg, batch_norm=False):
    # 卷积列表，依次
    layers = []
    # 输入通道
    in_channels = 3
    for v in cfg:
        # 是否遇到池化层
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 卷积操作，都是k=3*3,s=1,p=1
            conv = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1)
            # 是否加入归一化
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            # 重置输入
            in_channels = v
    print(layers)
    print(*layers)
    return nn.Sequential(*layers)


layer_configs = {
    # 这个参数配置表可以根据论文中表1得出。
    # 5个M表示五个池化，
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


# vgg11
def vgg11():
    return VGG(features=make_layers(layer_configs['A'], batch_norm=False))


# vgg11_bn
def vgg11_bn():
    return VGG(features=make_layers(layer_configs['A'], batch_norm=True))


# vgg13
def vgg13():
    return VGG(features=make_layers(layer_configs['B'], batch_norm=False))


# vgg13_bn
def vgg13_bn():
    return VGG(features=make_layers(layer_configs['B'], batch_norm=True))


# vgg16
def vgg16():
    return VGG(features=make_layers(layer_configs['C'], batch_norm=False))


# vgg16_bn
def vgg16_bn():
    return VGG(features=make_layers(layer_configs['C'], batch_norm=True))


# vgg19
def vgg19():
    return VGG(features=make_layers(layer_configs['E'], batch_norm=False))


# vgg19_bn
def vgg19_bn():
    return VGG(features=make_layers(layer_configs['E'], batch_norm=True))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = vgg19_bn()
print(vgg)
vgg = vgg.to(device)
summary(vgg, (3, 224, 224))
