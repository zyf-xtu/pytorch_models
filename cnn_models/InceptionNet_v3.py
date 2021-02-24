#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/18 14:38
# @Author  : zyf
# @File    : InceptionNet_v3.py
# @Software: PyCharm
from abc import ABC

import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models
# from torchviz import make_dot

'''
     CNN经典网络结构复现：LeNet5、AlexNet、VGG、ResNet、GoogleNet、InceptionNet等
     InceptionV3网络结构在InceptionV2版本上改进
     InceptionV3包含三种Inception模块，分别对应论文中的figure5,figure6,figure7图中的结构。
     这里表示一下三种Inception模块：
     figure5 中模块：InceptionmoduleA
                               filter concat
           /              /                   \                  \
          |              |                 3x3 conv               |                         
          |              |                     |                  |
        1x1 conv      5x5 conv             3x3 conv           1x1 conv
          |              |                     |                  |
          |           1x1 conv             1x1 conv        3x3 max pooling
           \              \                   /                  /
                                   base
        每个5×5卷积由两个3×3卷积替换.
        
     figure6 中模块：InceptionmoduleB 
                             filter concat
           /              /                   \                  \
          |              |                 nx1 conv               | 
          |              |                     |                  | 
          |              |                 1xn conv               | 
          |              |                     |                  | 
          |           nx1 conv             nx1 conv               |                         
          |              |                     |                  |
          |           1xn conv             1xn conv           1x1 conv
          |              |                     |                  |
       1x1 conv       1x1 conv             1x1 conv        3x3 max pooling
           \              \                   /                  /
                                   base
           n×n卷积分解后的Inception模块。在我们提出的架构中，对17×17的网格我们选择n=7                           
        
     figure7 中模块：InceptionmoduleC
                                 filter concat
           /         /         |          |         |           \
          |         |          |          |         |            | 
          |         |          |       1x3 conv  3x1 conv        |                         
          |         |          |          \        /             |
          |       1x3 conv  3x1 conv       3x3 conv           1x1 conv
          |          \        /               |                  |
       1x1 conv       1x1 conv             1x1 conv        3x3 max pooling
           \              \                   /                  /
                                   base
         具有扩展的滤波器组输出的InceptionmoduleC模块。这种架构被用于最粗糙的（8×8）网格，以提升高维表示，
         如第2节原则2所建议的那样。我们仅在最粗的网格上使用了此解决方案，因为这是产生高维度的地方，
         稀疏表示是最重要的，因为与空间聚合相比，局部处理（1×1 卷积）的比率增加.
                                   
     网络结构参数表：
     type            patch size/stride or remarks         input size
     conv                       3x3/2                     299x299x3
     conv                       3x3/1                     149x149x32
     conv padded                3x3/1                     147x147x32
     pool                       3x3/2                     147x147x64
     conv                       3x3/1                     73x73x64
     conv                       3x3/2                     71x71x80
     conv                       3x3/1                     35x35x192
     3xInceptionModuleA      As in figure5                35x35x288(涉及一个下采样InceptionModuleDownSmapeD的过程)
     5xInceptionModuleB      As in figure6                17x17x768(涉及一个下采样InceptionModuleDownSmapeE的过程)
     2xInceptionModuleC      As in figure7                8x8x1280
     pool                       8x8                       8x8x2048
     linear                     logits                    1x1x2048
     softmax                    classifier                1x1x1000     
 
'''


# 定义一个基础的卷积,一个卷积、BN层、激活ReLu
class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        """
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param kwargs: 将不定长的键值对，作为参数传递给一个函数，如果在一个函数里处理带名字的参数，你应该使用**kwargs
        """
        super(BasicConv2d, self).__init__()
        # 卷积
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, **kwargs)
        # BN层
        self.bn = nn.BatchNorm2d(out_channel)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    # 正向传播
    def forward(self, x):
        # 卷积
        x = self.conv(x)
        # bn
        x = self.bn(x)
        # relu
        x = self.relu(x)
        return x


# 定义Inception模块，基于figure5中的结构
class InceptionModuleA(nn.Module):
    def __init__(self, in_ch, out1x1, out5x5_reduce, out5x5, double3x3_reduce, double3x3, poolproj):
        """
        :param in_ch:输入的通道数
        :param out1x1:1x1分支的输出
        :param out5x5_reduce:5x5分支的第一个1x1卷积输出
        :param out5x5:5x5分支的输出
        :param double3x3_reduce:两个3x3的第一个1x1卷积输出
        :param double3x3:两个3x3后的输出
        :param poolproj:池化分支的输出
        """
        super(InceptionModuleA, self).__init__()
        # 第一个分支
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel=in_ch, out_channel=out1x1, kernel_size=1)
        )
        # 第二个分支5x5的
        self.branch2 = nn.Sequential(
            # 第一个1x1的卷积层
            BasicConv2d(in_channel=in_ch, out_channel=out5x5_reduce, kernel_size=1),
            # 第二个是3x3的卷积层，需要padding=1，保证输出的w*h不变
            BasicConv2d(in_channel=out5x5_reduce, out_channel=out5x5, kernel_size=5, padding=2)
        )
        # 第三个分支，两个3x3
        self.branch3 = nn.Sequential(
            # 第一个是1x1的卷积
            BasicConv2d(in_channel=in_ch, out_channel=double3x3_reduce, kernel_size=1),
            # 第二个是3x3的卷积
            BasicConv2d(in_channel=double3x3_reduce, out_channel=double3x3, kernel_size=3, padding=1),
            # 第三个也是3x3的卷积
            BasicConv2d(in_channel=double3x3, out_channel=double3x3, kernel_size=3, padding=1)
        )
        # 第四个分支
        self.branch4 = nn.Sequential(
            # 第一个是平均池化，kernel_size=3
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            # 第二个是1x1的卷积，目的是降低通道数
            nn.Conv2d(in_channels=in_ch, out_channels=poolproj, kernel_size=1)
        )

    # 前向传播
    def forward(self, x):
        bh1 = self.branch1(x)
        bh2 = self.branch2(x)
        bh3 = self.branch3(x)
        bh4 = self.branch4(x)
        # 将四个分支的结果，进行拼接,dim=1 表示在channel维度上进行，batchsize*channel*w*h
        out = [bh1, bh2, bh3, bh4]
        return torch.cat(out, dim=1)


# 定义InceptionModuleB模块，基于figure6中的结构
class InceptionModuleB(nn.Module):
    def __init__(self, in_ch, out1x1, out_nxn_reduce, out_nxn, double_nxn_reduce, double_nxn, poolproj):
        """
        :param in_ch: 输入通道数
        :param out1x1: 1x1分支的输出
        :param out_nxn_reduce: nxn分支的第一个1x1输出
        :param out_nxn: nxn分支的输出
        :param double_nxn_reduce: 两个nxn分支的第一个1x1输出
        :param double_nxn: 两个nxn分支的输出
        :param poolproj: 池化分支输出
        """
        super(InceptionModuleB, self).__init__()
        # 第一个分支
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel=in_ch, out_channel=out1x1, kernel_size=1)
        )
        # 第二个分支nxn,包含三个卷积层，依次是1x1、1x7、7x1的卷积核
        self.branch2 = nn.Sequential(
            # 第一个1x1的卷积层
            BasicConv2d(in_channel=in_ch, out_channel=out_nxn_reduce, kernel_size=1),
            # 第二个是1x7的卷积层，需要padding=(0,3)，保证输出的w*h不变
            BasicConv2d(in_channel=out_nxn_reduce, out_channel=out_nxn_reduce, kernel_size=(1, 7), padding=(0, 3)),
            # 第二个是7x1的卷积层，需要padding=(3,0)，保证输出的w*h不变
            BasicConv2d(in_channel=out_nxn_reduce, out_channel=out_nxn, kernel_size=(7, 1), padding=(3, 0))
        )
        # 第三个分支db_nxn,包含五个卷积，依次是1x1、1x7、7x1、1x7、7x1的卷积核
        self.branch3 = nn.Sequential(
            # 第一个1x1的卷积层
            BasicConv2d(in_channel=in_ch, out_channel=double_nxn_reduce, kernel_size=1),
            # 第二个是1x7的卷积层，需要padding=(0,3)，保证输出的w*h不变
            BasicConv2d(in_channel=double_nxn_reduce, out_channel=double_nxn_reduce, kernel_size=(1, 7),
                        padding=(0, 3)),
            # 第三个是7x1的卷积层，需要padding=(3,0)，保证输出的w*h不变
            BasicConv2d(in_channel=double_nxn_reduce, out_channel=double_nxn_reduce, kernel_size=(7, 1),
                        padding=(3, 0)),
            # 第四个是1x7的卷积层，需要padding=(0,3)，保证输出的w*h不变
            BasicConv2d(in_channel=double_nxn_reduce, out_channel=double_nxn_reduce, kernel_size=(1, 7),
                        padding=(0, 3)),
            # 第五个是7x1的卷积层，需要padding=(3,0)，保证输出的w*h不变
            BasicConv2d(in_channel=double_nxn_reduce, out_channel=double_nxn, kernel_size=(7, 1), padding=(3, 0))
        )
        # 第四个分支
        self.branch4 = nn.Sequential(
            # 第一个是平均池化，kernel_size=3
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            # 第二个是1x1的卷积，目的是降低通道数
            nn.Conv2d(in_channels=in_ch, out_channels=poolproj, kernel_size=1)
        )

    # 前向传播
    def forward(self, x):
        bh1 = self.branch1(x)
        bh2 = self.branch2(x)
        bh3 = self.branch3(x)
        bh4 = self.branch4(x)
        # 将四个分支的结果，进行拼接,dim=1 表示在channel维度上进行，batchsize*channel*w*h
        out = [bh1, bh2, bh3, bh4]
        return torch.cat(out, dim=1)


# 定义InceptionModuleC模块，基于figure7中的结构
class InceptionModuleC(nn.Module):
    def __init__(self, in_ch, out1x1, out_3x3_reduce, out_3x3, double_3x3_reduce, double_3x3, poolproj):
        super(InceptionModuleC, self).__init__()
        # 第一个分支
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel=in_ch, out_channel=out1x1, kernel_size=1)
        )
        # 第二个分支3x3,包含三个卷积层，分别是1x1、1x3、3x1的卷积核，其中在1x3和3x1的需要合并
        # 第一个1x1的卷积
        self.branch2_1x1 = BasicConv2d(in_channel=in_ch, out_channel=out_3x3_reduce, kernel_size=1)
        # 第二个是1x3的卷积层，需要padding=(0,1)，保证输出的w*h不变
        self.branch2_1x3 = BasicConv2d(in_channel=out_3x3_reduce, out_channel=out_3x3, kernel_size=(1, 3),
                                       padding=(0, 1))
        # 第二个是3x1的卷积层，需要padding=(1,0)，保证输出的w*h不变
        self.branch2_3x1 = BasicConv2d(in_channel=out_3x3_reduce, out_channel=out_3x3, kernel_size=(3, 1),
                                       padding=(1, 0))

        # 第三个分支double3x3,包含四个卷积层，分别是 1x1、3x3、1x3、3x1的卷积核，其中在1x3和3x1的需要合并
        # 第一个1x1的卷积
        self.branch3_1x1 = BasicConv2d(in_channel=in_ch, out_channel=double_3x3_reduce, kernel_size=1)
        # 第二个3x3的卷积，需要padding=1
        self.branch3_3x3 = BasicConv2d(in_channel=double_3x3_reduce, out_channel=double_3x3, kernel_size=3,
                                       padding=1)
        # 第三个是1x3的卷积层，需要padding=(0,1)，保证输出的w*h不变
        self.branch3_1x3 = BasicConv2d(in_channel=double_3x3, out_channel=double_3x3, kernel_size=(1, 3),
                                       padding=(0, 1))
        # 第四个是3x1的卷积层，需要padding=(1,0)，保证输出的w*h不变
        self.branch3_3x1 = BasicConv2d(in_channel=double_3x3, out_channel=double_3x3, kernel_size=(3, 1),
                                       padding=(1, 0))

        # 第四个分支
        self.branch4 = nn.Sequential(
            # 第一个是平均池化，kernel_size=3
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            # 第二个是1x1的卷积，目的是降低通道数
            nn.Conv2d(in_channels=in_ch, out_channels=poolproj, kernel_size=1)
        )

    # 前向传播
    def forward(self, x):
        bh1 = self.branch1(x)
        # 第二个分支，先经过一个1x1的卷积，后跟两个1x3和3x1的卷积
        bh2_1x1 = self.branch2_1x1(x)
        bh2_1x3 = self.branch2_1x3(bh2_1x1)
        bh2_3x1 = self.branch2_3x1(bh2_1x1)
        # 第二分支合并输出
        bh2 = torch.cat([bh2_1x3, bh2_3x1], dim=1)
        # 第三个分支，先经过一个1x1和3x3的卷积，后跟两个1x3和3x1的卷积
        bh3_1x1 = self.branch3_1x1(x)
        bh3_3x3 = self.branch3_3x3(bh3_1x1)
        bh3_1x3 = self.branch3_1x3(bh3_3x3)
        bh3_3x1 = self.branch3_3x1(bh3_3x3)
        # 第三分支合并输出
        bh3 = torch.cat([bh3_1x3, bh3_3x1], dim=1)
        # 第四个分支
        bh4 = self.branch4(x)
        # 将四个分支的结果，进行拼接,dim=1 表示在channel维度上进行，batchsize*channel*w*h
        out = [bh1, bh2, bh3, bh4]
        return torch.cat(out, dim=1)


# 定义第一个下采样InceptionModuleDownSmapeD模块,结构是基于figure5，类似于InceptionA的结构。
# 用于35x35 -> 17x17 下采样过程
# 该结构只有三个分支,1x1的分支取消了，只有3x3的分支和double3x3分支以及最大池化
# 该结构中stride=2，目的是进行下采样。
class InceptionModuleDownsmapeD(nn.Module):
    def __init__(self, in_ch, out3x3_reduce, out3x3, double3x3_reduce, double3x3):
        """
        :param in_ch:输入的通道数
        :param out1x1:1x1分支的输出
        :param out3x3_reduce:3x3分支的第一个1x1卷积输出
        :param out3x3:3x3分支的输出
        :param double3x3_reduce:两个3x3的第一个1x1卷积输出
        :param double3x3:两个3x3后的输出
        :param poolproj:池化分支的输出
        """
        super(InceptionModuleDownsmapeD, self).__init__()
        # 第一个分支
        self.branch1 = nn.Sequential(
            # 第一个是1x1的卷积
            BasicConv2d(in_channel=in_ch, out_channel=out3x3_reduce, kernel_size=1),
            # 第二个是3x3的卷积层,stride=2,保证了下采样
            BasicConv2d(in_channel=out3x3_reduce, out_channel=out3x3, kernel_size=3, stride=2)
        )
        # 第二个分支
        self.branch2 = nn.Sequential(
            # 第一个是1x1的卷积层
            BasicConv2d(in_channel=in_ch, out_channel=double3x3_reduce, kernel_size=1),
            # 第二个是3x3的卷积层
            BasicConv2d(in_channel=double3x3_reduce, out_channel=double3x3, kernel_size=3, padding=1),
            # 第三个也是3x3的卷积层,stride=2,保证了下采样
            BasicConv2d(in_channel=double3x3, out_channel=double3x3, kernel_size=3, stride=2)
        )
        # 第三个分支
        self.branch3 = nn.Sequential(
            # 3x3的kernel，stride=2
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

    # 前向传播
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        # 将三个结果进行拼接，dim=1表示在channel维度上进行，batchsize*channel*w*h
        out = [b1, b2, b3]
        return torch.cat(out, dim=1)


# 定义一个下采样模块：InceptionModuleDownsampleE,结构基于figure6，类似于InceptionB的结构
# 作用是17x17 -> 8x8  的下采样过程
# 参考源码部分
class InceptionModuleDownsampleE(nn.Module):
    def __init__(self, in_ch, out_3x3_reduce, out_3x3, double7x7_reduce, double_7x7):
        """
        :param in_ch: 输入通道数
        :param out_3x3_reduce: nxn分支的第一个1x1输出
        :param out_3x3: nxn分支的输出
        :param double_7x7_reduce: 两个nxn分支的第一个1x1输出
        :param double_7x7: 两个nxn分支的输出
        :param poolproj: 池化分支输出
        """
        super(InceptionModuleDownsampleE, self).__init__()
        # 第一个分支3x3的下采样,包含两个个卷积层，依次是1x1、3x3的卷积核
        self.branch1 = nn.Sequential(
            # 第一个1x1的卷积层
            BasicConv2d(in_channel=in_ch, out_channel=out_3x3_reduce, kernel_size=1),
            # 第二个是1x7的卷积层，需要padding=(0,3)，保证输出的w*h不变
            BasicConv2d(in_channel=out_3x3_reduce, out_channel=out_3x3, kernel_size=3, stride=2)
        )
        # 第二个分支db_7x7,包含四个卷积，依次是1x1、1x7、7x1、3x3的卷积核
        self.branch2 = nn.Sequential(
            # 第一个1x1的卷积层
            BasicConv2d(in_channel=in_ch, out_channel=double7x7_reduce, kernel_size=1),
            # 第二个是1x7的卷积层，需要padding=(0,3)，保证输出的w*h不变
            BasicConv2d(in_channel=double7x7_reduce, out_channel=double7x7_reduce, kernel_size=(1, 7),
                        padding=(0, 3)),
            # 第三个是7x1的卷积层，需要padding=(3,0)，保证输出的w*h不变
            BasicConv2d(in_channel=double7x7_reduce, out_channel=double7x7_reduce, kernel_size=(7, 1),
                        padding=(3, 0)),
            # 第四个是3x3的卷积层，stride=2,进行下采样
            BasicConv2d(in_channel=double_7x7, out_channel=double_7x7, kernel_size=3, stride=2)
        )
        # 第三个分支
        self.branch3 = nn.Sequential(
            # 3x3的kernel，stride=2
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

    # 前向传播
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        # 将三个结果进行拼接，dim=1表示在channel维度，batch*channel*w*h
        return torch.cat([b1, b2, b3], dim=1)


# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        :param in_channels:  输入的通道数
        :param num_classes:  分类数
        """
        super(InceptionAux, self).__init__()
        # 平均池化,输入：768x17x17  输出：768x5x5
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        # 第一个卷积，1x1的卷积  输入：768x5x5  输出：128x5x5
        self.conv1 = BasicConv2d(in_channel=in_channels, out_channel=128, kernel_size=1)
        # 第二个卷积，5x5的卷积 输入：128x5x5  输出：768x1x1
        self.conv2 = BasicConv2d(in_channel=128, out_channel=768, kernel_size=5)
        # 自适应池化，输入：768x1x1  输出：768x1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接操作
        self.fc = nn.Linear(in_features=768, out_features=num_classes)

    # 前向传播
    def forward(self, x):
        print(x.size())
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        # 展开
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x,dim=1)
        out = self.fc(x)
        return out


# 构建InceptionV3结构,参考源码参数
class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=False):
        """
        :param num_classes: 分类数
        """
        super(InceptionV3, self).__init__()
        self.num_classes = num_classes
        # 是否使用辅助分类器
        self.aux_logits = aux_logits
        # 常规卷积处理,卷积核：3,步长：2
        # 输入：3x299x299 输出：32x149x149
        self.conv1 = BasicConv2d(in_channel=3, out_channel=32, kernel_size=3, stride=2)
        # 常规卷积，输入：32x149x149  输出：32x147x147
        self.conv2 = BasicConv2d(in_channel=32, out_channel=32, kernel_size=3, stride=1)
        # 常规卷积，输入：32x147x147  输出：64x147x147,padding=1
        self.conv3 = BasicConv2d(in_channel=32, out_channel=64, kernel_size=3, stride=1, padding=1)
        # 池化操作,输入：64x147x147  输出：64x73x73 ceil_mode=True 表示向上取整，默认false向下取整
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # 常规卷积，输入：64x73x73  输出：80x73x73
        self.conv4 = BasicConv2d(in_channel=64, out_channel=80, kernel_size=1)
        # 常规卷积，输入：80x73x73  输出：192x71x71
        self.conv5 = BasicConv2d(in_channel=80, out_channel=192, kernel_size=3)
        # 池化操作，输入：192x71x71 输出：192x35x35
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 接下来是第一个Inception模块，包含三个块结构
        # 输入：192x35x35   输出：（64+64+96+32）256x35x35
        self.inception3a = InceptionModuleA(in_ch=192, out1x1=64, out5x5_reduce=48, out5x5=64, double3x3_reduce=64,
                                            double3x3=96, poolproj=32)
        # 输入：256x35x35   输出：64+64+96+64） 288 x 35 x 35
        self.inception3b = InceptionModuleA(in_ch=256, out1x1=64, out5x5_reduce=48, out5x5=64, double3x3_reduce=64,
                                            double3x3=96, poolproj=64)
        # 输入：288x35x35   输出：64+64+96+64） 288 x 35 x 35
        self.inception3c = InceptionModuleA(in_ch=288, out1x1=64, out5x5_reduce=48, out5x5=64, double3x3_reduce=64,
                                            double3x3=96, poolproj=64)

        # 第二个Inception模块，包含五个块结构
        # 首先第一个Inception模块要进行下采样操作，输入：288x35x35  输出：(384+96+288)768x17x17
        self.inception4a = InceptionModuleDownsmapeD(in_ch=288, out3x3_reduce=384, out3x3=384, double3x3_reduce=64,
                                                     double3x3=96)
        # 输入：768x17x17  输出： 768x17x17
        self.inception4b = InceptionModuleB(in_ch=768, out1x1=192, out_nxn_reduce=128, out_nxn=192,
                                            double_nxn_reduce=128, double_nxn=192, poolproj=192)
        # 输入：768x17x17  输出： 768x17x17
        self.inception4c = InceptionModuleB(in_ch=768, out1x1=192, out_nxn_reduce=160, out_nxn=192,
                                            double_nxn_reduce=160, double_nxn=192, poolproj=192)
        # 输入：768x17x17  输出： 768x17x17
        self.inception4d = InceptionModuleB(in_ch=768, out1x1=192, out_nxn_reduce=160, out_nxn=192,
                                            double_nxn_reduce=160, double_nxn=192, poolproj=192)
        # 输入：768x17x17  输出： 768x17x17
        self.inception4e = InceptionModuleB(in_ch=768, out1x1=192, out_nxn_reduce=192, out_nxn=192,
                                            double_nxn_reduce=192, double_nxn=192, poolproj=192)

        # 是否使用辅助分类器
        if self.aux_logits:
            self.Aux = InceptionAux(in_channels=768, num_classes=self.num_classes)

        # 第三个Inception模块，包含三个块结构
        # 首先第一个Inception模块要进行下采样操作，输入：768x17x17  输出：(320+192+768)1280x8x8
        self.inception5a = InceptionModuleDownsampleE(in_ch=768, out_3x3_reduce=192, out_3x3=320, double7x7_reduce=192,
                                                      double_7x7=192)
        # 输入：1280x8x8  输出： (320+(384+384)+(384+384)+192)2048x8x8
        self.inception5b = InceptionModuleC(in_ch=1280, out1x1=320, out_3x3_reduce=384, out_3x3=384,
                                            double_3x3_reduce=448, double_3x3=384, poolproj=192)
        # 输入：2048x8x8  输出： (320+(384+384)+(384+384)+192)2048x8x8
        self.inception5c = InceptionModuleC(in_ch=2048, out1x1=320, out_3x3_reduce=384, out_3x3=384,
                                            double_3x3_reduce=448, double_3x3=384, poolproj=192)

        # 自定义池化,用来固定输出的维度1*1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, self.num_classes)

    # 前向传播
    def forward(self, x):
        # Nx3x299x299
        x = self.conv1(x)
        # Nx32x149x149
        x = self.conv2(x)
        # Nx32x147x147
        x = self.conv3(x)
        # Nx64x147x147
        x = self.pool1(x)
        # Nx64x73x73
        x = self.conv4(x)
        # Nx80x73x73
        x = self.conv5(x)
        # Nx192x71x71
        x = self.pool2(x)
        # 开始Inception模块
        # Nx192x35x35
        x = self.inception3a(x)
        # Nx256x35x35
        x = self.inception3b(x)
        # Nx288x35x35
        x = self.inception3c(x)
        # Nx288x35x35
        x = self.inception4a(x)  # 进行下采样操作
        # Nx768x17x17
        x = self.inception4b(x)
        # Nx768x17x17
        x = self.inception4c(x)
        # Nx768x17x17
        x = self.inception4d(x)
        # Nx768x17x17
        x = self.inception4e(x)
        # 是否采用辅助分类器
        if self.aux_logits and self.training:
            aux = self.Aux(x)
        else:
            aux = None
        # Nx768x17x17
        x = self.inception5a(x)  # 进行下采样
        # Nx1280x8x8
        x = self.inception5b(x)
        # Nx2048x8x8
        x = self.inception5c(x)
        # Nx2048x8x8,自适应平均池化
        x = self.avg_pool(x)
        # Nx2048x1x1
        x = self.dropout(x)
        # 展开
        x = torch.flatten(x, 1)
        # x = x.view(x.size(0),-1)
        # Nx2048
        x = self.fc(x)
        return x, aux


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionV3()
model.to(device)
print(model)
X = torch.rand(2, 3, 299, 299)
X = X.to(device)
out, aux = model(X)
# g = make_dot(out)
# g.view()
print(out)
# print(out.size())
summary(model, (3, 299, 299))
