#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/15 16:15
# @Author  : zyf
# @File    : ResNet_50.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    CNN经典网络结构复现：LeNet5、AlexNet、VGG、ResNet、InceptionNet等
    ResNet50网络结构：50 = 1(conv1) + 3*3(第一个残差部分) +3*4(第二个残差部分) +3*6(第三个残差部分) +3*3(第四个残差部分) + 1(FC)
    需要设计一个残差块，ResBlock设计：
        该block中有三个卷积，分别是1x1,3x3,1x1，分别完成的功能就是维度下降，卷积，恢复维度。
        故bottleneck实现的功能就是对通道数进行压缩，再放大。
        这样设计的目的减少计算量和降低参数数目。
        包含三个个卷积层，每个卷积层后面跟一个归一化
        第一个卷积层是一个1*1的卷积核，是负责降维即channel的变化，不负责宽高的变化。
        第二个卷积层是一个3*3的卷积核，负责下采样，负责图像宽高的变化。
        第三个卷积层是一个1*1的卷积核，负责升维即channel的变化。
    第一部分卷积conv1：
         输入：224*224*3
         输出：112*112*64
         conv：kernel_size = 7*7 stride=2 padding=3
         输入：112*112*64
         输出：56*56*64
         max pooling : kernel_size =3 stride=2 padding=1
    第一个残差部分conv2：
        输入：56*56*64  输出：56*56*64
        包含3个个残差块，每个残差块里面有三个卷积层：1*1、3*3、1*1
    第二个残差部分conv2：
        输入：56*56*64  输出：28*28*128
        包含4个残差块，每个残差块里面有三个卷积层：1*1、3*3、1*1
        其中第一个残差块要做下采样
    第三个残差部分conv2：
        输入：28*28*128  输出：14*14*256
        包含6个残差块，每个残差块里面有三个卷积层：1*1、3*3、1*1
        其中第一个残差块要做下采样
    第四个残差部分conv2：
        输入：14*14*256  输出：7*7*512
        包含3个残差块，每个残差块里面有三个卷积层：1*1、3*3、1*1
        其中第一个残差块要做下采样
    自定义池化和全连接层
        avg_pool
        fc 
    注意：其实这部分的残差块与ResNet101、ResNet152的结构是一样的，不过是每个残差部分的数量不一致罢了，这里分开实现纯粹是为了代码熟练度。
    ResNet101 101= 1 + 3*3 + 3*4 +3*23 +3*3 + 1
    ResNet152 152= 1 + 3*8 + 3*4 +3*36 +3*3 + 1
'''


# 定义一个3*3的卷积
def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=padding,
                     bias=False)


# 定义一个1*1的卷积
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False)


# 残差块
class Bottleneck(nn.Module):
    # 每个残差结构的输出维度都是输入维度的4倍
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(Bottleneck, self).__init__()
        # 残差结构中的第一个卷积，1*1的卷积，负责channels的降低维度。
        self.conv1 = conv1x1(in_planes, out_planes)
        # 归一化层
        self.bn1 = nn.BatchNorm2d(out_planes)
        # 残差结构中的第二个卷积，3*3的卷积，进行downsample操作，负责图像宽高的变化
        self.conv2 = conv3x3(out_planes, out_planes, stride)
        # 归一化层
        self.bn2 = nn.BatchNorm2d(out_planes)
        # 残差结构中的第三个卷积：1*1的卷积，负责channels的升高维度。
        self.conv3 = conv1x1(out_planes, self.expansion * out_planes)
        # 归一化
        self.bn3 = nn.BatchNorm2d(self.expansion * out_planes)
        # 激活函数
        self.relu = nn.ReLU()
        # 重点部分，当残差块要进行downsample的时候，快捷连接也需要进行维度的同步，
        # 同步的方法是采用一个1*1的卷积，同时stride=2
        self.down_sample = None
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.down_sample = nn.Sequential(
                # 采用1*1的卷积进行维度同步 。下采样，W*H会变小。
                conv1x1(in_planes, self.expansion * out_planes, stride),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    # 前向传播
    def forward(self, x):
        # 恒等x
        identity = x
        # 残差块内的第一个卷积过程
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 残差块内的第二个卷积过程
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 残差块内的第二个卷积过程
        out = self.conv3(out)
        out = self.bn3(out)

        # 右侧快捷连接计算
        if self.down_sample is not None:
            identity = self.down_sample(x)

        # 残差相加部分，将输出结果相加
        out += identity
        # 然后使用relu激活函数
        out = self.relu(out)
        return out


# ResNet50主体
class ResNet50(nn.Module):
    def __init__(self, nums=1000):
        super(ResNet50, self).__init__()
        # 分类数
        self.nums = nums
        # 第一个卷积部分，这部分没有涉及残差模块，标准的卷积过程
        self.conv1 = nn.Sequential(
            # 卷积层，输入：224*224*3  输出：112*112*64
            # 计算方式 w=(w+2*p-k)/2 + 1 h=(h+2*p-k)/2 +1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            # 批量归一化
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 池化，输入：112*112*64  输出：56*56*64
            # 计算方式：w=(w+2*p-k)/2 + 1 h=(h+2*p-k)/2 +1 下采样需要向下取整
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第一个残差部分包含3个残差块，9个卷积层
        self.conv2_x = nn.Sequential(
            # 第一个block要进行下downsample的，因为in_planes != Bottleneck.expansion*out_planes
            Bottleneck(in_planes=64, out_planes=64),
            # 第一个block的out_planes =64 * Bottleneck.expansion，第二个block的 in_planes =64 * Bottleneck.expansion
            Bottleneck(in_planes=64 * Bottleneck.expansion, out_planes=64),
            Bottleneck(in_planes=64 * Bottleneck.expansion, out_planes=64)
        )
        # 第二个残差部分包含4个残差块，12个卷积层
        self.conv3_x = nn.Sequential(
            # 上一个残差部分的输出的channels是：64 * Bottleneck.expansion
            # 所以第一个block的in_planes = 64 * Bottleneck.expansion ,out_planes = 128 * Bottleneck.expansion
            # 同时也进行downsample操作
            Bottleneck(in_planes=64 * Bottleneck.expansion, out_planes=128, stride=2),
            # 第一个block的out_planes：128 * Bottleneck.expansion 是第二个block的in_planes=128 * Bottleneck.expansion
            Bottleneck(in_planes=128 * Bottleneck.expansion, out_planes=128),
            Bottleneck(in_planes=128 * Bottleneck.expansion, out_planes=128),
            Bottleneck(in_planes=128 * Bottleneck.expansion, out_planes=128)
        )
        # 第三个残差部分包含6个残差块，18个卷积层
        self.conv4_x = nn.Sequential(
            # 上一个残差部分的输出的channels是：128 * Bottleneck.expansion
            # 所以第一个block的in_planes = 128 * Bottleneck.expansion , out_planes= 256 * Bottleneck.expansion
            # 同时也进行downsample操作
            Bottleneck(in_planes=128 * Bottleneck.expansion, out_planes=256, stride=2),
            # 第一个block的out_planes：256 * Bottleneck.expansion 是第二个block的in_planes=256 * Bottleneck.expansion
            Bottleneck(in_planes=256 * Bottleneck.expansion, out_planes=256),
            Bottleneck(in_planes=256 * Bottleneck.expansion, out_planes=256),
            Bottleneck(in_planes=256 * Bottleneck.expansion, out_planes=256),
            Bottleneck(in_planes=256 * Bottleneck.expansion, out_planes=256),
            Bottleneck(in_planes=256 * Bottleneck.expansion, out_planes=256)
        )
        # 第四个残差部分包含3个残差块，9个卷积层
        self.conv5_x = nn.Sequential(
            # 上一个残差部分的输出channels是：256 * Bottleneck.expansion
            # 第一个block的in_planes= 256 * Bottleneck.expansion，out_planes= 512*  Bottleneck.expansion
            # 同时也进行downsample
            Bottleneck(in_planes=256 * Bottleneck.expansion, out_planes=512, stride=2),
            # 第一个block的out_planes：512 * Bottleneck.expansion 是第二个block的in_planes=512 * Bottleneck.expansion
            Bottleneck(in_planes=512 * Bottleneck.expansion, out_planes=512),
            Bottleneck(in_planes=512 * Bottleneck.expansion, out_planes=512)
        )
        # 自定义池化层，用来固定输出的size大小
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 定义全连接层,输出是类别数
        self.fc = nn.Linear(512 * Bottleneck.expansion, self.nums)

    def forward(self, x):
        # 卷积层
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        # 自定义池化，固定输出大小
        out = self.avg_pool(out)
        # 将数据展开的两种方式
        out = out.view(out.size(0),-1)
        # out = torch.flatten(out, 1)
        # 全连接层
        out = self.fc(out)
        return out

x = torch.rand((2,3,224,224))
res50 = ResNet50()
print(res50)
out = res50(x)
print(out)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
res50 = res50.to(device)
summary(res50, (3, 224, 224))
