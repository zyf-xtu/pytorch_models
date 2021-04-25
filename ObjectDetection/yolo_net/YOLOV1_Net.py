#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/1 17:25
# @Author  : zyf
# @File    : YOLOV1_Net.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

"""
    复现YOLOV1论文中的神经网络结构。
    YOLO系列的目标检测，一个基于卷积神经网络的目标检测算法，将目标检测的分类问题转化为回归问题。
    区别基于滑动窗口和候选框的两阶目标检测算法（two stage）,YOLO是一阶目标检测（one stage）的鼻祖。
    YOLO的优点：
    1.速度快，one stage detection的开山之作
    2.YOLO可以很好的避免背景的错误
    3.泛化能力强
    缺点：
    1.检测的精度低于当前最好的检测模型，最大的劣势就是精确度低。
    2.定位不够精确
    3.对小物体的检测效果不好，尤其是密集型小物体，因为一个栅格只能预测两个物体。
    4.召回率低
    YOLOV1论文中的figure3中网络结构复现（自己琢磨参数）
    YOLO借鉴了GoogleNet分类网络结构，不同的是YOLO使用1x1卷积层和3x3卷积层替代了Inception Module，
    包含24个卷积层和2个全连接层。其中卷积层用来提取图像特征，全连接层用来预测图像位置和类别概率。
    输入尺寸：3x448x448
"""


# 定义一个卷积类，包含卷积，归一化，激活函数
class Layer(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super(Layer, self).__init__()
        # 卷积
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, **kwargs)
        # 批量归一化
        self.bn = nn.BatchNorm2d(out_ch)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    # 前向传播
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 主干网络
class YoloModel(nn.Module):
    def __init__(self):
        super(YoloModel, self).__init__()
        # 输入尺寸3x448x448
        # 卷积层,第一部分卷积,一个卷积层和一个池化层
        self.conv_layer1 = nn.Sequential(
            # 输入：3x448x448     输出：64x223x223
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 池化，输入：64x223x223  输出：64x112x112
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        # 第二部分卷积，一个卷积层和一个池化
        self.conv_layer2 = nn.Sequential(
            # 输入：64x112x112     输出：192x112x112
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # 池化，输入：192x112x112  输出：192x56x56
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        # 第三部分卷积，四个卷积层和一个池化层
        self.conv_layer3 = nn.Sequential(
            # 第一个卷积层是1x1的卷积，out_channel=128
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 第二个卷积是3x3的卷积，out_channel=256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 第三个卷积层是1x1的卷积，out_channel=256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 第四个卷积是3x3的卷积，out_channel=512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 最大池化，输入：512x56x56  输出：512x28x28
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第四部分卷积，包含十个卷积，一个池化,其中重复的卷积有八个
        # 其中1x1和3x3的卷积重复4次
        block4 = []
        block4.append(Layer(in_ch=512, out_ch=256, kernel_size=1))
        block4.append(Layer(in_ch=256, out_ch=512, kernel_size=3, padding=1))
        layers4 = []
        for i in range(4):
            layers4.extend(block4)
        layers4.append(Layer(in_ch=512, out_ch=512, kernel_size=1))
        layers4.append(Layer(in_ch=512, out_ch=1024, kernel_size=3, padding=1))
        # 最大池化，输入：512x28x28  输出：512x14x14
        layers4.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer4 = nn.Sequential(*layers4)
        # 第五部分卷积，包含6个卷积，最后一个卷积stride=2
        layers5 = []
        layers5.append(Layer(in_ch=1024, out_ch=512, kernel_size=1))
        layers5.append(Layer(in_ch=512, out_ch=1024, kernel_size=3, padding=1))
        layers5.append(Layer(in_ch=1024, out_ch=512, kernel_size=1))
        layers5.append(Layer(in_ch=512, out_ch=1024, kernel_size=3, padding=1))
        layers5.append(Layer(in_ch=1024, out_ch=1024, kernel_size=3, padding=1))
        # 1024x14x14 -> 1024x7x7
        layers5.append(Layer(in_ch=1024, out_ch=1024, kernel_size=3, stride=2, padding=1))
        self.conv_layer5 = nn.Sequential(*layers5)
        # 第六部分卷积，包含两个卷积
        self.conv_layer6 = nn.Sequential(
            Layer(in_ch=1024, out_ch=1024, kernel_size=3, padding=1),
            Layer(in_ch=1024, out_ch=1024, kernel_size=3, padding=1),
        )
        # 两个全连接层
        # 自定义池化，目的固定输出size
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 30 * 7 * 7)
        )

    # 前向传播
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        # 数据展开
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # 增加sigmoid函数是为了将输出全部映射到(0,1)之间
        x = torch.sigmoid(x)
        # 输出为7x7x30的tensor
        # x = x.reshape(-1, 7, 7, 30)
        x = x.view(-1, 7, 7, 30)
        return x


x = torch.rand((1, 3, 448, 448))
model = YoloModel()
print(model)
out = model(x)
print(out)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
summary(model, (3, 448, 448))
