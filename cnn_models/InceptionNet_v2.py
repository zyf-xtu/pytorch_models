#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/18 21:11
# @Author  : zyf
# @File    : InceptionNet_v2.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models
# from torchviz import make_dot
'''
    CNN经典网络结构复现：LeNet5、AlexNet、VGG、ResNet、GoogleNet、InceptionNet等
    InceptionV2版本相比InceptionV1版本改进如下：
    1.使用了Batch Normalization ，加快模型的训练速度以及梯度消失和爆炸的问题。
    2.使用了两个3x3的网络结构代替了5x5的网络结构，降低了参数数量，并减轻了过拟合。
    3.由于使用了BN,可以增大学习速率，加快学习衰减速度。
    Inception V2模块结构：
        第一个1x1分支：只有1x1卷积
        第二个3x3分支：两个卷积层，首先经过一个1x1的卷积，后跟一个3x3的卷积
        第三个3x3分支（在V1里这是一个5x5的卷积）：三个卷积层，首先一个1x1的卷积，后跟两个3x3的卷积
        第四个pool分支：首先经过pool层，后跟一个1x1的卷积。

                                 特征拼接
           /              /                   \                  \
          |              |                 3x3 conv               |                         
          |              |                     |                  |
        1x1 conv      3x3 conv             3x3 conv           1x1 conv
          |              |                     |                  |
          |           1x1 conv             1x1 conv        3x3 max pooling
           \              \                   /                  /
                                 上一层
        
        四个分支，分别做卷积，然后拼接输出。
    
    V2版本的网络结构相比V1版本的网络结构：
        在第一个分支branch1上不做改变
        在第二个分支branch2上先经过一个1x1的卷积层，然后再经过3x3的卷积层。
        在第三个分支branch3上也要先经过一个1x1的卷积层，然后用两个3x3的卷积代替了5x5的卷积。
        在第四个分支branch4上先经过一个3x3的max pooling ,然后再使用1x1的卷积层进行降维。
    对应参数表：
    type      patch size/stride     output size     depth     #1x1     #3x3 reduce     #3x3     #double #3x3 reduce     double#3x3     Pool+proj
    convolution*     7x7/2          112x112x64        1
    max pool         3x3/2          56x56x64          0
    convolution      3x3/1          56x56x192         1                    64           192
    max pool         3x3/2          28x28x192         0
    inception(3a)                   28x28x256         3        64          64           64              64                  96           avg+32
    inception(3b)                   28x28x320         3        64          64           96              64                  96           avg+64
    inception(3c)    stride=2       28x28x576         3        0           128          160             64                  96           max+pass through(downsample)
    inception(4a)                   14x14x576         3        224         64           96              96                  128          avg+128
    inception(4b)                   14x14x576         3        192         96           128             96                  128          avg+128
    inception(4c)                   14x14x576         3        160         128          160             128                 160          avg+128
    inception(4d)                   14x14x576         3        96          128          192             160                 192          avg+128
    inception(4e)    stride=2       14x14x1024        3        0           128          192             192                 256          max+pass through(downsample)
    inception(5a)                   7x7x1024          3        352         192          320             160                 224          avg+128
    inception(5b)                   7x7x1024          3        352         192          320             192                 224          avg+128
    avg pool          7x7/1         1x1x1024          0        

'''


# 定义一个基础的卷积类,包含一个卷积层、BN层和一个ReLu激活层，正向传播函数
class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        """
        :param in_channel:输入通道
        :param out_channel:输出通道
        :param kwargs:**kwargs 允许你将不定长度的键值对, 作为参数传递给一个函数。 如果你想要在一个函数里处理带名字的参数, 你应该使用**kwargs
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


# 定义Inception模块
class InceptionModule(nn.Module):
    def __init__(self, in_channel, out1x1, out3x3_reduce, out3x3, db3x3_reduce, db3x3, pool_proj):
        super(InceptionModule, self).__init__()
        """
        :param in_channel: 输入的深度
        :param out1x1:第一个分支输出
        :param out3x3_reduce: 第二个分支的第一个1x1卷积输出
        :param out3x3:第二个分支的输出
        :param db3x3_reduce:第三个分支的第一个1x1卷积输出
        :param db3x3:第三个分支经过两个3x3之后的输出
        :param pool_proj:第四个分支的输出
        """
        # 第一个分支
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel=in_channel, out_channel=out1x1, kernel_size=1)
        )
        # 第二个分支
        self.branch2 = nn.Sequential(
            # 第一个是1x1的卷积层
            BasicConv2d(in_channel=in_channel, out_channel=out3x3_reduce, kernel_size=1),
            # 第二个是3x3的卷积层，需要padding=1，保证输出的w*h不变
            BasicConv2d(in_channel=out3x3_reduce, out_channel=out3x3, kernel_size=3, padding=1)
        )
        # 第三个分支
        self.branch3 = nn.Sequential(
            # 第一个是1x1的卷积层
            BasicConv2d(in_channel=in_channel, out_channel=db3x3_reduce, kernel_size=1),
            # 第二个是3x3的卷积层
            BasicConv2d(in_channel=db3x3_reduce, out_channel=db3x3, kernel_size=3, padding=1),
            # 第三个也是3x3的卷积层
            BasicConv2d(in_channel=db3x3, out_channel=db3x3, kernel_size=3, padding=1)
        )
        # 第四个分支
        self.branch4 = nn.Sequential(
            # 第一个是平均池化,3x3的kernel
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            # 第二个是1x1的卷积,作用是降维
            nn.Conv2d(in_channels=in_channel, out_channels=pool_proj, kernel_size=1)
        )

    # 前向传播
    def forward(self, x):
        bh1 = self.branch1(x)
        bh2 = self.branch2(x)
        bh3 = self.branch3(x)
        bh4 = self.branch4(x)
        # 将四个分支的结果，进行拼接,dim=1 表示在channel维度上进行。batchsize*channel*width*height
        out = [bh1, bh2, bh3, bh4]
        return torch.cat(out, dim=1)


# 定义带有下采样的Inception模块（inception3c和inception4e）
# 该结构只有三个分支,1x1的分支取消了，只有3x3的分支和double3x3分支以及最大池化
# 该结构中stride=2，目的是进行下采样。
class InceptionDownsample(nn.Module):
    def __init__(self, in_channel, out3x3_reduce, out3x3, db3x3_reduce, db3x3):
        super(InceptionDownsample, self).__init__()
        """
                :param in_channel: 输入的深度
                :param out3x3_reduce: 第一个分支的第一个1x1卷积输出
                :param out3x3:第一个分支的输出
                :param db3x3_reduce:第二个分支的第一个1x1卷积输出
                :param db3x3:第二个分支经过两个3x3之后的输出
                :param pool_proj:第四个分支的输出
                """
        # 第一个分支，也就是3x3的分支
        self.branch1 = nn.Sequential(
            # 第一个是1x1的卷积层
            BasicConv2d(in_channel=in_channel, out_channel=out3x3_reduce, kernel_size=1),
            # 第二个是3x3的卷积层
            BasicConv2d(in_channel=out3x3_reduce, out_channel=out3x3, kernel_size=3, stride=2, padding=1)
        )
        # 第二个分支
        self.branch2 = nn.Sequential(
            # 第一个是1x1的卷积层
            BasicConv2d(in_channel=in_channel, out_channel=db3x3_reduce, kernel_size=1),
            # 第二个是3x3的卷积层
            BasicConv2d(in_channel=db3x3_reduce, out_channel=db3x3, kernel_size=3, padding=1),
            # 第三个也是3x3的卷积层,stride=2,保证了下采样
            BasicConv2d(in_channel=db3x3, out_channel=db3x3, kernel_size=3, stride=2, padding=1)
        )
        # 第三个个分支，最大池化
        self.branch3 = nn.Sequential(
            # 最大池化,3x3的kernel，stride=2
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

    # 前向传播
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = [b1, b2, b3]
        return torch.cat(concat, dim=1)


# 定义InceptionV2
class InceptionV2(nn.Module):
    def __init__(self, num_class=1000):
        super(InceptionV2, self).__init__()
        # 分类数
        self.num_class = num_class
        # 第一个常规卷积，卷积核:7x7 步长：2 填充:3
        # 输入：3x224x224  输出：64x112x112 (224 + 6 -7)/2 +1 = 112
        self.conv1 = BasicConv2d(in_channel=3, out_channel=64, kernel_size=7, stride=2, padding=3)
        # 最大池化，输入：64x112x112    输出：64x56x56
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # ceil_mode=True 表示向上取整，默认是false向下取整
        # 第二个常规卷积1x1，输入：64x56x56  输出：64x56x56
        self.conv2 = BasicConv2d(in_channel=64, out_channel=64, kernel_size=1)
        # 第三个常规卷积，输入：64x56x56  输出：192x56x56
        self.conv3 = BasicConv2d(in_channel=64, out_channel=192, kernel_size=3, stride=1, padding=1)
        # 第四个最大池化，输入：192x56x56  输出：192x28x28
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # 接下来是Inception模块,将表格中对应的参数放进去就行了
        # inception3a输入：N*192*28*28 -> N*256*28*28
        self.inception3a = InceptionModule(in_channel=192, out1x1=64, out3x3_reduce=64, out3x3=64, db3x3_reduce=64,
                                           db3x3=96, pool_proj=32)
        # inception3b 输入：N*256*28*28 -> N*320*28*28
        # 256=64+64+96+32 -> 320=64+96+96+64  以下都是类似的计算
        self.inception3b = InceptionModule(in_channel=256, out1x1=64, out3x3_reduce=64, out3x3=96, db3x3_reduce=64,
                                           db3x3=96, pool_proj=64)
        # inception3c 该模块是进行下采样，stride=2
        # 输入： N*320*28*28  输出： N*576*14*14
        self.inception3c = InceptionDownsample(in_channel=320, out3x3_reduce=128, out3x3=160, db3x3_reduce=64, db3x3=96)
        # inception4a 输入：N*576*14*14  输出：N*576*14*14
        self.inception4a = InceptionModule(in_channel=576, out1x1=224, out3x3_reduce=64, out3x3=96, db3x3_reduce=96,
                                           db3x3=128, pool_proj=128)
        # inception4b 输入：N*576*14*14  输出：N*576*14*14
        self.inception4b = InceptionModule(in_channel=576, out1x1=192, out3x3_reduce=96, out3x3=128, db3x3_reduce=96,
                                           db3x3=128, pool_proj=128)
        # inception4c 输入：N*576*14*14  输出：N*576*14*14
        self.inception4c = InceptionModule(in_channel=576, out1x1=160, out3x3_reduce=128, out3x3=160, db3x3_reduce=128,
                                           db3x3=160, pool_proj=96)
        # inception4d 输入：N*576*14*14  输出：N*576*14*14
        self.inception4d = InceptionModule(in_channel=576, out1x1=96, out3x3_reduce=128, out3x3=192, db3x3_reduce=160,
                                           db3x3=192, pool_proj=96)
        # inception4e 该模块是进行下采样，stride=2
        # 输入： 输入：N*576*14*14  输出：N*(192+256+576=1024)*7*7
        self.inception4e = InceptionDownsample(in_channel=576, out3x3_reduce=128, out3x3=192, db3x3_reduce=192,
                                              db3x3=256)
        # inception5a 输入：N*1024*7*7  输出：N*1024*7*7
        self.inception5a = InceptionModule(in_channel=1024, out1x1=352, out3x3_reduce=192, out3x3=320, db3x3_reduce=160,
                                           db3x3=224, pool_proj=128)
        # inception5b 输入：输入：N*1024*7*7  输出：N*1024*7*7
        self.inception5b = InceptionModule(in_channel=1024, out1x1=352, out3x3_reduce=192, out3x3=320, db3x3_reduce=192,
                                           db3x3=224, pool_proj=128)
        # 自定义池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # fc
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_class)
        )

    # 前向传播
    def forward(self, x):
        # 常规卷积
        out = self.conv1(x)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.max_pool2(out)
        # inception3模块
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.inception3c(out)
        # inception4模块
        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        out = self.inception4e(out)
        # inception5模块
        out = self.inception5a(out)
        out = self.inception5b(out)
        # 自定义池化
        out = self.avg_pool(out)
        # 展平操作
        out = torch.flatten(out, 1)
        # fc
        out = self.fc(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionV2()
model.to(device)
print(model)
X = torch.rand(2, 3, 224, 224)
X = X.to(device)
out = model(X)
# g = make_dot(out)
# g.view()
print(out)
print(out.size())
summary(model, (3, 224, 224))
