#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 14:42
# @Author  : zyf
# @File    : ResNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    单独实现了ResNet18、34和50，具体实现方式请看具体文件
    这里需要对ResNet系列的模型进行组装。
    通过参数、模块化来实现不同的层的ResNet。
    代码结构：
    主体结构：ResNet类
    浅层的ResNet18、34的残差块结构：BasicBlock
    比较深层的ResNet50、101、152的残差块结构：BottleBlock
    层计算方式：
    ResNet18 18= 1 + 2*2 + 2*2 +2*2 +2*2 + 1
    ResNet34 34= 1 + 2*3 + 2*4 +2*6 +2*3 + 1 
    ResNet101 101= 1 + 3*3 + 3*4 +3*6 +3*3 + 1 
    ResNet101 101= 1 + 3*3 + 3*4 +3*23 +3*3 + 1 
    ResNet152 152= 1 + 3*3 + 3*8 +3*36 +3*3 + 1
'''


# 定义一个3*3的卷积
def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=padding,
                     bias=False)


# 定义一个1*1的卷积
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False)


# 浅层的残差块,包含两个卷积层
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        # 第二个卷积
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    # 前向传播
    def forward(self, x):
        # 右侧恒等式
        identity = x

        # 残差块第一个卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 残差块第二个卷积
        out = self.conv2(out)
        out = self.bn2(out)
        # 由于网络传播中会发生下采样，当进行下采样的时候，为保持维度一致，快捷连接部分也要进行维度同步
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差相加
        out += identity
        # 激活函数
        out = self.relu(out)
        return out


# 深层的残差块,包含三个卷积层
class Bottleneck(nn.Module):
    # 每个残差结构的输出维度都是输入维度的4倍
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
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
        self.relu = nn.ReLU(inplace=True)
        # 是否发生下采样
        self.downsample = downsample

    def forward(self, x):
        # 恒等x
        identity = x
        # 残差块内第一个卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 残差块内的第二个卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 残差块内的第三个卷积
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果进行了下采样，要进行维度的同步
        if self.downsample is not None:
            identity = self.downsample(x)
        # 残差相加
        out += identity
        # 相加后激活函数relu
        out = self.relu(out)
        return out


# 残差结构的主体
class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=1000):
        super(ResNet, self).__init__()
        # 分类数
        self.num_class = num_class
        # 初始化输入channels
        self.in_planes = 64
        # 所有残差网络的第一个卷积部分都是一样的,卷积、归一化、激活、池化
        # 输入：224*224*3  输出：112*112*64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # 归一化
        self.bn1 = nn.BatchNorm2d(64)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        # 池化，输入：112*112*64  输出：56*56*64
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 开始残差块部分,第一部分残差
        self.layer1 = self.__make_layer(block, 64, layers[0])
        # 第二部分残差,需要下采样，所以stride=2
        self.layer2 = self.__make_layer(block, 128, layers[1], stride=2)
        # 第三部分残差，需要下采样，所以stride=2
        self.layer3 = self.__make_layer(block, 256, layers[2], stride=2)
        # 第四部分残差，需要下采样，所以stride=2
        self.layer4 = self.__make_layer(block, 512, layers[3], stride=2)
        # 自定义池化，固定输出尺寸
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层,输入是：512*block.expansion  输出是：分类数
        self.fc = nn.Linear(512 * block.expansion, self.num_class)

    # 定义卷积的层数
    def __make_layer(self, block, out_planes, blocks, stride=1):
        """
        :param block: 残差块类型
        :param in_planes:输入channels
        :param out_planes:输出channels
        :param stride:步长，决定是否进行下采样
        :return:
        """
        downsample = None
        # 当步长为2的时候需要进行下采样，快捷连接也需要维度同步
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, out_planes * block.expansion, stride),
                nn.BatchNorm2d(out_planes * block.expansion)
            )
        # 残差块内的卷积层
        layers = []
        layers.append(block(self.in_planes, out_planes, stride, downsample))
        self.in_planes = out_planes * block.expansion
        # 第一个残差块负责下采样，所以从第二个残差块开始
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一部分卷积、归一化、激活函数和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # 第一部分残差
        x = self.layer1(x)
        # 第二部分残差
        x = self.layer2(x)
        # 第三部分残差
        x = self.layer3(x)
        # 第四部分残差
        x = self.layer4(x)

        # 自定义池化，固定输出size
        x = self.avg_pool(x)
        # 全连接前，需要将维度展开
        x = torch.flatten(x, 1)
        # 全连接层
        x = self.fc(x)

        return x


# ResNet18
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# ResNet34
def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


# ResNet50
def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


# ResNet101
def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


# ResNet152
def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


resnet = resnet50()
print(resnet)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)
summary(resnet, (3, 224, 224))
