#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 9:32
# @Author  : zyf
# @File    : GoogleNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

'''
    CNN经典网络结构复现：LeNet5、AlexNet、VGG、ResNet、GoogleNet、InceptionNet等
    GoogleNet V1版本使用的是Inception V1模块,是在Inception版本上升级，加入了1x1的卷积层，目的是减少参数和计算量。
    该版本比naive inception多了三个1x1的卷积层
        在第一个分支branch1上不做改变
        在第二个分支branch2上先经过一个1x1的卷积层，然后再经过3x3的卷积层。
        在第三个分支branch3上也要先经过一个1x1的卷积层，然后再经过5x5的卷积层。
        在第四个分支branch4上先经过一个3x3的max pooling ,然后再使用1x1的卷积层进行降维。
    Inception V1模块结构：
        
                                 特征拼接
           /              /                   \                  \
        1x1 conv      3x3 conv             5x5 conv        1x1 conv
          |              |                     |                  |
          |           1x1 conv             1x1 conv        3x3 max pooling
           \              \                   /                  /
                                 上一层
                                 
        四个分支，分别做卷积，然后拼接输出。
    GoogleNet类
    
'''


# 定义一个基础的卷积类，包含一个卷积层和一个ReLu激活层，正向传播函数
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        :param in_channels: 输入channels
        :param out_channels: 输出的channels
        :param kwargs: **kwargs 允许你将不定长度的键值对, 作为参数传递给一个函数。 如果你想要在一个函数里处理带名字的参数, 你应该使用**kwargs
        """
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)  # inplace-选择是否进行覆盖运算

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# 定义Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        """
        :param in_channels: 输入的深度
        :param ch1x1: 第一个分支的1x1的输出
        :param ch3x3_reduce: 第二个分支的1x1卷积的输出
        :param ch3x3:第二个分支的3x3的输出
        :param ch5x5_reduce:第三个分支的1x1的输出
        :param ch5x5:第三个分支的5x5的输出
        :param pool_proj:第四个分支的输出
        """
        super(Inception, self).__init__()
        # 第一个分支
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 第二个分支
        self.branch2 = nn.Sequential(
            # 第一个是1x1的卷积层
            BasicConv2d(in_channels, ch3x3_reduce, kernel_size=1),
            # 第二个是3x3的卷积层，需要padding=1，保证输出的w*h不变
            BasicConv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )
        # 第三个分支
        self.branch3 = nn.Sequential(
            # 第一个是1x1的卷积层，目的是降维
            BasicConv2d(in_channels, ch5x5_reduce, kernel_size=1),
            # 第二个是5x5的卷积层，需要padding=2，保证输出的w*h不变
            BasicConv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2)
        )
        # 第四个分支
        self.branch4 = nn.Sequential(
            # 首先要经过一个3x3的池化
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # 然后经经过1x1的卷积，进行降维
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    # 正向传播
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # 将四个分支的结果，进行拼接,dim=1 表示在channel维度上进行。batchsize*channel*width*height
        out = [b1, b2, b3, b4]
        return torch.cat(out, dim=1)


# 辅助分类器设计
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        # 输入：in_channels*14*14   输出：in_channels*4*4
        # self.avg_pool = nn.AvgPool2d(kernel_size=5,stride=3)
        # 或者试用自定义池化，固定输出尺寸
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        # 经过1x1的卷积，进行降维
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        # 定义一个relu
        self.relu = nn.ReLU(inplace=True)
        # 定义一个dropout
        self.dropout = nn.Dropout(0.5)
        # 第一个全连接 128*4*4
        self.fc1 = nn.Linear(2048, 1024)
        # 第二个全连接
        self.fc2 = nn.Linear(1024, num_classes)

    # 正向传播
    def forward(self, x):
        # 辅助分类器aux1 是从inception(4a)处分支：N*512*14*14
        # 辅助分类器aux2 是从inception(4d)处分支：N*528*14*14
        x = self.avg_pool(x)
        # aux1:N*512*4*4   aux2:N*528*4*4
        # 使用1x1的卷积层进行降维到128
        print('aux+++', x.size())
        x = self.conv(x)
        print('aux----',x.size())
        # N*128*4*4
        x = torch.flatten(x, 1)
        x = self.relu(x)
        # N*2048
        x = self.fc1(x)
        # 使用nn.functional里面的函数
        x = F.dropout(x, 0.5, training=self.training)
        # x = self.dropout(x)
        # N*1024
        x = self.fc2(x)
        # N*1000(num_classes)
        return x


# GoogleNet主体类设计
class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=False, init_weights=None):
        """
        :param num_classes: 分类数
        :param aux_logits: 是否采用辅助分类器，默认是
        :param init_weights: 初始化权重
        """
        super(GoogleNet, self).__init__()
        # 分类数
        self.num_classes = num_classes
        # 是否采用辅助分类器
        self.aux_logits = aux_logits
        # 输入：N*3*224*224   输出：N*64*112*112
        self.conv1 = BasicConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        # ceil_mode=True 表示向上取整，默认是false向下取整
        # 输入：N*64*112*112  输出：N*64*56*56
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 输入：N*64*56*56  输出：N*64*56*56
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        # 输入：N*64*56*56  输出：N*192*56*56
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        # 池化输入：N*192*56*56  输出：N*192*28*28
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 接下来是Inception模块,将表格中对应的参数放进去就行了
        # 输入：N*192*28*28 -> N*256*28*28
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # 输入：N*256*28*28 -> N*480*28*28
        # 256=64+128+32+32 -> 480=128+192+96+64  以下都是类似的计算
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # 最大池化 N*480*28*28 -> N*480*14*14
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # N*480*14*14 -> N*512*14*14
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # N*512*14*14 -> N*512*14*14
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # N*512*14*14 -> N*512*14*14
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # N*512*14*14 -> N*528*14*14
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # N*528*14*14 -> N*832*14*14
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        # 最大池化 N*832*14*14 -> N*832*7*7
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # N*832*7*7 -> N*832*7*7
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        # N*832*7*7 -> N*832*7*7
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 是否使用辅助分类器
        if aux_logits:
            self.aux1 = InceptionAux(512, self.num_classes)
            self.aux2 = InceptionAux(528, self.num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        # 自定义池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 使用dropout
        self.dropout = nn.Dropout(0.5)
        # 全连接输出
        self.fc = nn.Linear(1024, self.num_classes)

        # 初始化权重
        # if init_weights:
        #     self._init_weights()

    # 前向传播
    def forward(self, x):
        # N*3*224*224
        x = self.conv1(x)
        # N*64*112*112
        x = self.max_pool1(x)
        print('pool1',x.size())
        # N*64*56*56
        x = self.conv2(x)
        # N*64*56*56
        x = self.conv3(x)
        # N*192*56*56
        x = self.max_pool2(x)
        print('pool2', x.size())
        # N*192*28*28
        x = self.inception3a(x)
        print('3a', x.size())
        # N*256*28*28
        x = self.inception3b(x)
        # N*480*28*28
        x = self.max_pool3(x)
        # N*480*14*14
        x = self.inception4a(x)

        # N*512*14*14
        # 是否使用辅助分类器 同时是否是训练模式
        if self.aux1 is not None and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N*512*14*14
        x = self.inception4c(x)
        # N*512*14*14
        x = self.inception4d(x)

        # N*528*14*14
        # 是否使用辅助分类器 同时是否是训练模式
        if self.aux2 is not None and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N*832*14*14
        x = self.max_pool4(x)
        # N*832*7*7
        x = self.inception5a(x)
        # N*832*7*7
        x = self.inception5b(x)
        # N*1024*7*7
        x = self.avg_pool(x)
        # 展平操作
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        print(x.size())
        # N*1024
        x = self.dropout(x)
        # 全连接
        x = self.fc(x)
        # N*1000(num_classes)
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x

    # 初始化权重
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)



googlenet = GoogleNet()
print(googlenet)
X = torch.rand(2,3,224,224)
out = googlenet(X)
print(out)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
googlenet = googlenet.to(device)
summary(googlenet, (3, 224, 224))