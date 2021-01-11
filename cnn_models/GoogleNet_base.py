#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/21 10:01
# @Author  : zyf
# @File    : GoogleNet_base.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    CNN经典网络结构复现：LeNet5、AlexNet、VGG、ResNet、GoogleNet、InceptionNet等
    GoogleNet 实现一个最初的版本。
    GoogleNet使用的是Inception模块，最初的版本结构为一个1*1、3*3、5*5和3*3的四个分支，没有使用1*1的卷积降维。
    由于是简单复现最初GoogleNet的结构，所以这里没有辅助分类器。
    Inception模块
        结构如下：
                                特征拼接
           /             /                   \                  \
        1x1 conv      3x3 conv             5x5 conv        3x3 max pooling
           \             \                   /                  /
                                上一层
        四个分支，分别做卷积，然后拼接输出。
        该特征图先被复制成4份并分别被传至接下来的4个部分。
        例如：输入：32*32*256
        我们假设这4个部分对应的滑动窗口的步长均为1，其中，
        1×1卷积层的Padding为0,要求输出的特征图深度为128；
        3×3卷积层的Padding为1，要求输出的特征图深度为192；
        5×5卷积层的Padding为2，要求输出的特征图深度为96；
        3×3最大池化层的 Padding为1，输出的特征深度不变256
        分别得到这4部分输出的特征图为32×32×128、32×32×192、32×32×96 和 32×32×256，
        最后在合并层进行合并，得到32×32×672的特征图，合并的方法是将各个部分输出的特征图相加，
        最后这个Naive Inception单元输出的特征图维度是32×32×672，
        总的参数量就是1*1*256*128+3*3*256*192+5*5*256*96=1089536
    GoogleNet类
    
'''


# Naive Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3, n5x5):
        '''
        :param in_channels:
        :param n1x1: 第一个分支的out_channels
        :param n3x3: 第二个分支的out_channels
        :param n5x5: 第三个分支的out_channels
        :param pool: 第四个分支是maxpooling
        '''
        super(Inception, self).__init__()
        # 1x1convolutions branch
        self.branch1 = nn.Sequential(
            # padding=0 保证输出的h*w不变
            nn.Conv2d(in_channels=in_channels, out_channels=n1x1, kernel_size=1, stride=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )
        # 3x3convolutions branch
        self.branch2 = nn.Sequential(
            # padding=1 保证输出的h*w不变
            nn.Conv2d(in_channels=in_channels, out_channels=n3x3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )
        # 5x5convolutions branch
        self.branch3 = nn.Sequential(
            # padding=2 保证输出的h*w不变
            nn.Conv2d(in_channels=in_channels, out_channels=n5x5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )
        # 3x3 max pooling branch
        self.branch4 = nn.Sequential(
            # padding=1 保证输出的h*w不变
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )

    # 前向传播
    def forward(self, x):
        # 第一个分支结果
        b1 = self.branch1(x)
        print('b1 :', b1.size())
        # 第二个分支结果
        b2 = self.branch2(x)
        print('b2 :', b2.size())
        # 第三个分支结果
        b3 = self.branch3(x)
        print('b3 :', b3.size())
        # 第四个分支结果
        b4 = self.branch4(x)
        print('b4 :', b4.size())
        # 将Inception模块的四个分支输出进行拼接，dim=1 表示在channel维度上进行。batchsize*channel*width*height
        out = torch.cat((b1, b2, b3, b4), dim=1)
        print(out.size())
        return out


# 实现Naive GoogleNet
class GoogleNet(nn.Module):
    def __init__(self, num_class=1000):
        super(GoogleNet, self).__init__()
        self.num_class = num_class
        # 第一部分卷积，没有涉及Inception模块
        # 输入：3*224*224  输出：16*28*28
        self.conv1 = nn.Sequential(
            # 第一个卷积层，输入：3*224*224 输出：8*112*112
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # 池化层：输入：8*112*112 输出：8*56*56
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 第二个卷积 输入：8*56*56 输出：8*56*56
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1),

            # 第三个卷积 输入：8*56*56  输出：16*56*56
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            # 池化 输入：16*56*56   输出：16*28*28
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        '''
            ------------注意这里只是想复现naive inception的结构----------------
            每个inception的输入参数都是自定义的。
            naive inception的一个缺点是网络的深度会一直增加
            所以在前三个分支中设置的参数都很小，第四个分支是max pooling在channels上是原样不变的
            所以计算公式很简单。假如上一层输入是32，branch1=16，branch2=8，branch3=8
            out = 16+8+8+32 = 64
        '''
        # inception3a 模块 16*28*28
        self.inception3a = Inception(16, 8, 16, 8)
        # inception3b 模块 输入：8+16+8+16 = 48*28*28
        self.inception3b = Inception(48, 16, 32, 16)
        # 最大池化模块
        self.max_pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # inception4a 模块 输入：16+32+16+48 = 112*14*14
        self.inception4a = Inception(112, 16, 32, 16)
        # inception4b 模块 输入：16+32+16+112 = 176*14*14
        self.inception4b = Inception(176, 16, 32, 16)
        # inception4c 模块 输入：16+32+16+176 = 240*14*14
        self.inception4c = Inception(240, 32, 64, 32)
        # inception4d 模块 输入：32+64+32+240 = 368*14*14
        self.inception4d = Inception(368, 32, 64, 32)
        # inception4e 模块 输入：32+64+32+368 = 496*14*14
        self.inception4e = Inception(496, 32, 64, 32)
        # 这里会有一个最大池化层 输入：32+64+32+496 = 624*14*14
        self.max_pooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # inception5a 模块 输入：32+64+32+496 = 624*7*7
        self.inception5a = Inception(624, 64, 128, 32)
        # inception5b 模块 输入：32+64+32+624 = 848*7*7
        self.inception5b = Inception(848, 64, 128, 64)
        # 自定义池化 输入：32+64+32+848 = 1104*7*7
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(1104, self.num_class)

    def forward(self, x):
        x = self.conv1(x)
        # 3a、3b两个inception模块
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pooling1(x)
        # 4a、4b、4c、4d、4e五个inception模块
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.max_pooling2(x)
        # 5a、5b两个inception模块
        x = self.inception5a(x)
        x = self.inception5b(x)
        # 自定义池化
        x = self.avg_pool(x)
        print(x.size())
        # 展平操作
        x = torch.flatten(x, 1)
        # 全连接层
        x = self.fc(x)
        return x

x = torch.rand((2,3,224,224))
googlenet = GoogleNet()
print(googlenet)
out = googlenet(x)
print(out)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
googlenet = googlenet.to(device)
summary(googlenet, (3, 224, 224))
