#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/17 14:48
# @Author  : zyf
# @File    : LeNet5.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    经典CNN网络结构复现：LeNet5、AlexNet、VGG、ResNet、InceptionNet等
    复现LeNet5网络结构
    LeNet5的网络结构，根据结构来构建模型，主要结构如下
    1.卷积层C1
    2.下采样层S1
    3.卷积层C2
    4.下采样层S2
    5.全连接层F5
    6.全连接层F6
    7.输出层F7
'''


class LeNet5(nn.Module):
    def __init__(self, nums=10):
        super(LeNet5, self).__init__()
        self.nums = nums  # 分类数，默认是模mnist的十分类
        # 定义卷积层及结构
        self.conv = nn.Sequential(
            # 卷积层C1，输入：32*32*1  输出：28*28*6
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            # 下采样S2，输入：28*28*6  输出：14*14*6
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 激活函数
            nn.Sigmoid(),
            # 卷积层C3 输入：14*14*6  输出：10*10*16
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            # 下采样S4 输入：10*10*16  输出：5*5*16
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 激活函数
            nn.Sigmoid()
        )
        # 定义全连接层及结构
        self.fc = nn.Sequential(
            # 全连接层F5 输入为：5*5*16 输出：120
            nn.Linear(in_features=5 * 5 * 16, out_features=120),
            # 全连接层F6 输入：120  输出：84
            nn.Linear(in_features=120, out_features=84),
            # 输出层F7,输入：84 输出：nums
            nn.Linear(in_features=84, out_features=self.nums)
        )

    def forward(self, x):
        x = self.conv(x)
        # 需要将多维度的值展平为一维，送入linear中，但是需要保持batchsize的维度
        # 例如2*512*7*7 变成2*25088
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 测试数据
x = torch.rand((2, 1, 32, 32))
lenet = LeNet5()
print(lenet)
out = lenet(x)
print(out)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lenet = lenet.to(device)
# 网络模型的数据流程及参数信息
summary(lenet,(1,32,32))