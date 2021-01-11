#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/11 16:34
# @Author  : zyf
# @File    : ResNet_18.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

'''
    CNN经典网络结构复现：LeNet5、AlexNet、VGG、ResNet、InceptionNet等
    ResNet18网络结构：18 = 1(conv1) + 2*2(第一个残差部分) +2*2(第二个残差部分) +2*2(第三个残差部分) +2*2(第四个残差部分) + 1(FC)
    需要设计一个残差块，ResBlock设计：
        包含两个卷积层，每个卷积层后面跟一个归一化
        kernel_size = 3 卷积核大小
        stride不固定，目的是为了降采样，保证残差的维度与真正输出的维度一致

    第一部分卷积conv1：
         输入：224*224*3
         输出：112*112*64
         conv：kernel_size = 7*7 stride=2 padding=3
         
         输入：112*112*64
         输出：56*56*64
         max pooling : kernel_size =3 stride=2 padding=1
         
    第一个残差部分conv2：
        输入：56*56*64  输出：56*56*64
        包含两个残差块，每个残差块里面有两个卷积层
    
    第二个残差部分conv2：
        输入：56*56*64  输出：28*28*128
        包含两个残差块，每个残差块里面有两个卷积层，
        其中第一个残差块要做下采样
    第三个残差部分conv2：
        输入：28*28*128  输出：14*14*256
        包含两个残差块，每个残差块里面有两个卷积层
        其中第一个残差块要做下采样
    第四个残差部分conv2：
        输入：14*14*256  输出：7*7*512
        包含两个残差块，每个残差块里面有两个卷积层
        其中第一个残差块要做下采样
    自定义池化和全连接层
        avg_pool
        fc 
    
    注意：其实这部分的残差块与ResNet18的结构是一样的，不过是每个残差部分的数量不一致罢了，这里分开实现纯粹是为了代码熟练度。
    ResNet18 18= 1 + 2*2 + 2*2 +2*2 +2*2 + 1
    ResNet34 34= 1 + 2*3 + 2*4 +2*6 +2*3 + 1     
'''


# 设计18和34残差块，ResNet18和ResNet34 用的3*3的卷积，而且每个残差块都只有两层卷积
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        # 残差块内的第一个卷积，当stride！=1时，要进行下采样downsample
        # 例如56*56*64 -> 28*28*128 的时候要进行downsample,这时候要stride=2
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                               padding=1)
        # 卷积后跟的bn层
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 激活函数ReLu
        self.relu = nn.ReLU(inplace=True)
        # 残差块内的第二个卷积，k=3,s=1,p=1,这个卷积层没什么变化，in_channels和out_channels 是一样的
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        # 第二个bn层
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 快捷连接设计，也就是右边x的部分，在做残差相加的时候，必须保证残差的维度与真正输出的维度相等（注意这里维度是宽高以及深度）
        self.shortcut = None
        print(in_channel, out_channel, stride)
        # 重点部分，当残差块要进行downsample的时候，快捷连接也需要进行维度的同步，
        # 同步的方法是采用一个1*1的卷积，同时stride=2
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                # 采用1*1的卷积进行维度同步 。下采样，W*H会变小 。
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    # 前向传播
    def forward(self, x):
        # 残差块的右边x
        identity = x
        # 残差块计算流程
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 快捷连接计算的结果
        if self.shortcut is not None:
            identity = self.shortcut(x)
        # 两个结果相加
        out += identity
        out = self.relu(out)
        return out


# 设计ResNet网络结构
class ResNet(nn.Module):
    def __init__(self, nums=1000):
        super(ResNet, self).__init__()
        # 分类数
        self.nums = nums
        # 第一部分卷积conv1 输入：224*224*3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第一个残差部分，包含两个残差块，由于没有涉及残差维度变化，两个残差块都是一样的
        self.conv2 = nn.Sequential(
            ResBlock(in_channel=64, out_channel=64),
            ResBlock(in_channel=64, out_channel=64)
        )
        # 第二个残差部分，包含两个残差块，四个卷积层
        self.conv3 = nn.Sequential(
            # 第一个残差块需要进行下采样，必须保证残差的维度与真正输出的维度相等（注意这里维度是宽高以及深度）
            ResBlock(in_channel=64, out_channel=128, stride=2),
            ResBlock(in_channel=128, out_channel=128)
        )
        # 第三个残差部分，包含两个残差块，四个卷积层
        self.conv4 = nn.Sequential(
            # 第一个残差块需要进行下采样，必须保证残差的维度与真正输出的维度相等（注意这里维度是宽高以及深度）
            ResBlock(in_channel=128, out_channel=256, stride=2),
            ResBlock(in_channel=256, out_channel=256)
        )
        # 第四个残差部分，包含两个残差块，四个卷积层
        self.conv5 = nn.Sequential(
            # 第一个残差块需要进行下采样，必须保证残差的维度与真正输出的维度相等（注意这里维度是宽高以及深度）
            ResBlock(in_channel=256, out_channel=512, stride=2),
            ResBlock(in_channel=512, out_channel=512)
        )
        # 自定义池化层，用来固定输出的size大小
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 定义全连接层,输出是类别数
        self.fc = nn.Linear(512, self.nums)

    # 前向传播
    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # 自定义池化，固定输出大小
        x = self.avg_pool(x)
        # 将特征向量展开
        x = torch.flatten(x, 1)
        # 全连接层
        x = self.fc(x)
        return x

x = torch.rand((2,3,224,224))
res = ResNet()
print(res)
out = res(x)
print(out)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
res = res.to(device)
summary(res, (3, 224, 224))
