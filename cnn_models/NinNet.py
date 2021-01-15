#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/13 10:37
# @Author  : zyf
# @File    : NinNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

"""
    CNN经典网络结构复现：LeNet5、AlexNet、VGG、ResNet、InceptionNet等
    本文实现NinNet 全称：Network in Network
    该网络是2014年发表的，主要有两个创新点：
    1.使用了MLP Convolution layers 卷积层
        传统的CNN的卷积滤波器是底层数据块的广义线性模型（generalized linear model）GLM
        在这篇论文中使用了MLPConv 代替了GLM
        MLPConv 是在常规卷积后面接入若干个1x1卷积，每个特征图视为一个神经元特征图。
        特征图通过通过多个1x1的卷积就类似多个神经元线性组合，这样就像是MLP(多层感知器)了
    
    2.Global Average Pooling (全局平均池化)
        论文中提出使用全局平均池化代替全连接层，对最后一层的特征图进行全局平均池化，得到的结果向量直接输入到softmax层。
        全局平均池取代完全连通层上的一个优点是，通过增强特征映射和类别之间的对应关系，它更适合于卷积结构。
        因此，特征映射可以很容易地解释为类别信任映射。另一个优点是在全局平均池中没有优化参数，从而避免了这一层的过度拟合。
        此外，全局平均池综合了空间信息，从而对输入的空间平移具有更强的鲁棒性。
    重点关注：该文章是第一个使用1x1卷积的，可以实现跨通道特征融合和通道的升维降维，减少网络参数
    
    网络结构参考网上给定的参数，原论文没有给出参数：
    网络包括三个mlpconv层的nin模块和一个全局平均池化，在每个mlpconv层中，有一个三层感知器。
    第一个nin模块：
        第一个常规卷积输入：3x224x224 kernel_size = 11 ,output=96,stride=4 输出：96x54x54
        第二个多层感知器输入：96x54x54 kernel_size = 1 输出：96x54x54
        第三个多层感知器输入：96x54x54 kernel_size = 1 输出：96x54x54
    后跟一个maxpool kernel_size= 3 stride=2  输入：96x54x54   输出：96x26x26
    第二个nin模块：
        第一个常规卷积输入：96x26x26 kernel_size = 5 ,output=256,padding=2,stride=1 输出：256x26x26
        第二个多层感知器输入：256x26x26 kernel_size = 1 输出：256x26x26
        第三个多层感知器输入：256x26x26 kernel_size = 1 输出：256x26x26
    后跟一个maxpool kernel_size= 3 stride=2  输入：256x26x26  输出：256x12x12
    第三个nin模块：
        第一个常规卷积输入：256x12x12 kernel_size = 3 ,output=384,padding=1,stride=1 输出：384x12x12
        第二个多层感知器输入：384x12x12 kernel_size = 1 输出：384x12x12
        第三个多层感知器输入：384x12x12 kernel_size = 1 输出：384x12x12
    后跟一个maxpool kernel_size= 3 stride=2  输入：384x12x12  输出：384x5x5
    第四个nin模块：
        第一个常规卷积输入：384x5x5 kernel_size = 3 ,output=num_class(分类数),padding=1,stride=1 输出：num_classx5x5
        第二个多层感知器输入：num_classx5x5 kernel_size = 1 输出：num_classx5x5
        第三个多层感知器输入：num_classx5x5 kernel_size = 1 输出：num_classx5x5
    全局平均池化
        AdaptiveAvgPool2d((1,1))    
"""


# 定义nin模块
def nin_block(in_channel, out_channel, kernel_size, stride, padding=0):
    """
    :param in_channel: 输入通道
    :param out_channel: 输出通道
    :param kernel_size: 卷积核大小
    :param stride: 步长
    :param padding: 填充
    :return:
    """
    blk = nn.Sequential(
        # 第一个卷积是常规卷积
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        # 激活函数
        nn.ReLU(),
        # 1x1卷积
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1),
        # 激活函数
        nn.ReLU(),
        # 1x1卷积
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1),
        # 激活函数
        nn.ReLU()
    )
    return blk


# 展开

# NinNet网络结构
class NinNet(nn.Module):
    def __init__(self, num_class=1000):
        super(NinNet, self).__init__()
        self.num_class = num_class
        # 第一个nin模块输入：3x224x224   输出：96x54x54
        self.nin1 = nin_block(in_channel=3, out_channel=96, kernel_size=11, stride=4)
        # 第一个最大池化，输入：96x54x54  输出：96x26x26
        self.max_pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        # 第二个nin模块，输入：96x26x26  输出：256x26x26
        self.nin2 = nin_block(in_channel=96,out_channel=256,kernel_size=5,stride=1,padding=2)
        # 第二个最大池化：输入：256x26x26  输出：256x12x12
        self.max_pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        # 第三个nin模块，输入：256x12x12   输出：384x12x12
        self.nin3 = nin_block(in_channel=256,out_channel=384,kernel_size=3,stride=1,padding=1)
        # 第三个最大池化，输入：384x12x12   输出：384x5x5
        self.max_pool3 = nn.MaxPool2d(kernel_size=3,stride=2)
        # 第四个nin模块：输入：384x5x5     输出：num_classx5x5
        self.nin4 = nin_block(in_channel=384,out_channel=self.num_class,kernel_size=3,stride=1,padding=1)
        # 加上一个dropout层
        self.dropout = nn.Dropout(0.5)
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        # nin模块
        x = self.nin1(x)
        # 最大池化
        x = self.max_pool1(x)
        # nin模块
        x = self.nin2(x)
        # 最大池化
        x = self.max_pool2(x)
        # nin模块
        x = self.nin3(x)
        # 最大池化
        x = self.max_pool3(x)
        # nin模块
        x = self.nin4(x)
        # 全局平均池化GAP
        x = self.gap(x)
        # 展开
        x = x.view(x.size(0),-1)

        return x


model = NinNet()
print(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = torch.rand((2,3,224,224))
x = x.to(device)
out = model(x)
print(out.size())
summary(model,(3,224,224))