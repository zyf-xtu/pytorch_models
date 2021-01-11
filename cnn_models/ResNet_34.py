#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 17:19
# @Author  : zyf
# @File    : ResNet_34.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    CNN经典网络结构复现：LeNet5、AlexNet、VGG、ResNet、InceptionNet等
    ResNet34网络结构：34 = 1(conv1) + 2*3(第一个残差部分) +2*4(第二个残差部分) +2*6(第三个残差部分) +2*3(第四个残差部分) + 1(FC)
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
        包含3个个残差块，每个残差块里面有两个卷积层
    第二个残差部分conv2：
        输入：56*56*64  输出：28*28*128
        包含4个残差块，每个残差块里面有两个卷积层，
        其中第一个残差块要做下采样
    第三个残差部分conv2：
        输入：28*28*128  输出：14*14*256
        包含6个残差块，每个残差块里面有两个卷积层
        其中第一个残差块要做下采样
    第四个残差部分conv2：
        输入：14*14*256  输出：7*7*512
        包含3个残差块，每个残差块里面有两个卷积层
        其中第一个残差块要做下采样
    自定义池化和全连接层
        avg_pool
        fc 
    注意：其实这部分的残差块与ResNet18的结构是一样的，不过是每个残差部分的数量不一致罢了，这里分开实现纯粹是为了代码熟练度。
    ResNet18 18= 1 + 2*2 + 2*2 +2*2 +2*2 + 1
    ResNet34 34= 1 + 2*3 + 2*4 +2*6 +2*3 + 1
'''


# 定义一个3*3的卷积核
def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=padding)


# 定义一个1*1的卷积核
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 设计残差块
class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResBlock, self).__init__()
        # 残差块内的第一个卷积，当stride！=1时，要进行下采样downsample，使分辨率降低，即高和宽减半，同时会让深度随之增加。
        # 例如56*56*64 -> 28*28*128 的时候要进行downsample,这时候要stride=2
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        # 批量归一化
        self.bn1 = nn.BatchNorm2d(out_planes)
        # 激励函数
        self.relu = nn.ReLU()
        # 残差块内的第二个卷积，没有下采样的操作
        self.conv2 = conv3x3(out_planes, out_planes)
        # 批量归一化
        self.bn2 = nn.BatchNorm2d(out_planes)
        # 重点部分，当残差块要进行downsample的时候，快捷连接也需要进行维度的同步，
        # 同步的方法是采用一个1*1的卷积，同时stride=2
        self.down_sample = None
        if stride != 1 or in_planes != out_planes:
            self.down_sample = nn.Sequential(
                # 采用1*1的卷积进行维度同步 。下采样，W*H会变小 。
                conv1x1(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes)
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

        # 右侧快捷连接计算
        if self.down_sample is not None:
            identity = self.down_sample(x)

        # 残差相加部分，将输出结果相加
        out += identity
        # 然后使用relu激活函数
        out = self.relu(out)
        return out


# 主干网络
class ResNet34(nn.Module):
    def __init__(self, nums=1000):
        super(ResNet34, self).__init__()
        # 网络的分类数
        self.nums = nums
        # 第一个卷积部分，这部分没有涉及残差模块，标准的卷积过程
        self.conv1 = nn.Sequential(
            # 卷积层，输入：224*224*3  输出：112*112*64
            # 计算方式 w=(w+2*p-k)/2 + 1 h=(h+2*p-k)/2 +1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            # 批量归一化
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 池化，输入：112*112*64  输出：56*56*64
            # 计算方式：w=(w+2*p-k)/2 + 1 h=(h+2*p-k)/2 +1 下采样需要向下取整
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第一个残差部分包含3个残差块，6个卷积层，该部分没有进行下采样，输入与输出维度相同
        # 输入：56*56*64  输出：56*56*64
        self.conv2_x = nn.Sequential(
            ResBlock(in_planes=64, out_planes=64),
            ResBlock(in_planes=64, out_planes=64),
            ResBlock(in_planes=64, out_planes=64)
        )
        # 第二个残差部分包含4个残差块，8个卷积层，该部分的第一个残差块的第一个卷积层要进行下采样，stride=2进行维度的变换
        # 第一个残差块的输入：56*56*64 经过第一个卷积时候进行downsample后的输出：28*28*128
        self.conv3_x = nn.Sequential(
            ResBlock(in_planes=64,out_planes=128,stride=2),
            ResBlock(in_planes=128,out_planes=128),
            ResBlock(in_planes=128, out_planes=128),
            ResBlock(in_planes=128, out_planes=128)
        )
        # 第三个残差部分包含6个残差块，12个卷积层，该部分的第一个残差块内的第一个卷积要进行下采样，stride=2进行维度变换
        # 第一个残差块的输入：28*28*128 经过第一个卷积stride=2的downsam后输出为：14*14*256
        self.conv4_x = nn.Sequential(
            ResBlock(in_planes=128,out_planes=256,stride=2),
            ResBlock(in_planes=256,out_planes=256),
            ResBlock(in_planes=256, out_planes=256),
            ResBlock(in_planes=256, out_planes=256),
            ResBlock(in_planes=256, out_planes=256),
            ResBlock(in_planes=256, out_planes=256)
        )
        # 第四个残差部分包含3个残差块，6个卷积层，该部分的第一个残差块内的第一个卷积要进行下采样，stride=2进行维度变换
        # 第一个残差块的输入：14*14*256 经过第一个卷积stride=2的downsam后输出为：7*7*512
        self.conv5_x = nn.Sequential(
            ResBlock(in_planes=256, out_planes=512, stride=2),
            ResBlock(in_planes=512, out_planes=512),
            ResBlock(in_planes=512, out_planes=512)
        )
        # 设置自定义池化，用来固定输出的维度1*1
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # 全连接层，输出即为类别数
        self.fc = nn.Linear(1*1*512,self.nums)

    def forward(self,x):
        # 第一个卷积的输出
        out = self.conv1(x)
        # 第一个残差块的输出
        out = self.conv2_x(out)
        # 第二个残差块的输出
        out = self.conv3_x(out)
        # 第三个残差块的输出
        out = self.conv4_x(out)
        # 第四个残差块的输出
        out = self.conv5_x(out)
        # 自定义池化，固定输出大小
        out = self.avg_pool(out)
        # 将数据维度展开
        out = out.view(out.size(0),-1)
        # out = torch.flatten(x, 1)
        # 全连接层，输出
        out = self.fc(out)
        return out


x = torch.rand((2,3,224,224))
res34 = ResNet34()
print(res34)
out = res34(x)
print(out)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
res34 = res34.to(device)
summary(res34,(3,224,224))
