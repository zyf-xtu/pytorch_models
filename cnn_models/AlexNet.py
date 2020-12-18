#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 18:04
# @Author  : zyf
# @File    : AlexNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    经典CNN网络结构复现：LeNet5、AlexNet、VGG、ResNet、InceptionNet等
    复现AlexNet网络结构
    根据AlexNet的网络结构来构建网络模型，由于在2012年受到硬件条件的限制，模型结构是设计在
    两个GPU上的并行计算。现在我们将其简化到一个GPU上。网络结构没有变化，还是8层，卷积5层
    全连接3层，使用了ReLu和Dropout。
    主要结构如下：
    卷积部分
        1.卷积层C1 有ReLu和MaxPooling
        2.卷积层C2 有ReLu和MaxPooling
        3.卷积层C3 有ReLu
        4.卷积层C4 有ReLu
        5.卷积层C5 有ReLu和MaxPooling
    全连接部分
        6.全连接F6
        7.全连接F7
        8.全连接F8(out)
'''


# AlexNet的结构类
class AlexNet(nn.Module):
    def __init__(self, nums):
        super(AlexNet, self).__init__()
        # 分类数
        self.nums = nums
        # 卷积部分
        self.conv = nn.Sequential(
            # 卷积层C1  输入：224*224*3  输出：54*54*96
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            # 池化层，输入：54*54*96 输出：26*26*96
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 卷积层C2 输入：26*26*96    输出：26*26*256
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # 最大池化层，输入：26*26*256  输出：12*12*256
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 卷积层C3 输入：12*12*256   输出：12*12*384  没有pooling
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 卷积层C4 输入：12*12*384   输出：12*12*256  没有pooling
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 卷积层C5 输入：12*12*256   输出：12*12*256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 池化层  输入：12*12*256    输出：5*5*256
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 增加一个自定义池化层，固定输出size
        self.avg_pool = nn.AdaptiveAvgPool2d((5, 5))
        # 全连接部分
        self.fc = nn.Sequential(
            # 全连接FC1 输入：5*5*256 -> 4096
            nn.Linear(in_features=5 * 5 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            # 全连接FC2 4096 -> 4096
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            # 全连接FC3 4096 -> nums(1000)
            nn.Linear(in_features=4096, out_features=1000)
        )

    def forward(self, x):
        x = self.conv(x)
        # 自定义池化层
        x = self.avg_pool(x)
        # 将维度展开，但是要保持batchsize的维度
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.fc(x)
        return x


# 测试数据
x = torch.rand((2, 3, 227, 227))
alex = AlexNet(1000)
print(alex)
out = alex(x)
print(out)
# 是否使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alex = alex.to(device)
# 网络模型数据流程及参数信息
summary(alex, (3, 227, 227))
