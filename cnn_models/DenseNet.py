#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 20:19
# @Author  : zyf
# @File    : DenseNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

# from torchviz import make_dot

'''
     CNN经典网络结构复现：LeNet5、AlexNet、VGG、ResNet、GoogleNet、InceptionNet等
     本程序复现DenseNet网络（稠密连接网络）
     DenseNet与ResNet的思路很像，主要区别在于DenseNet是建立前面所有的层与后面层的密集连接（dense connection）,
     在跨层的短连接上，ResNet使用的相加操作，DenseNet使用的是连接操作。
     主要优点：
     缓解了梯度弥散和消失的问题
     增加了特征传播
     促进了特征的重用
     本质上减少了参数的数量
     DenseNet的网络结构
     dense_layer 卷积块，DenseNet 使用了ResNet改良版的'批量归一化、激活、卷积'的结构
     denseblock 稠密模块
     transition_layer模块（过渡层）
     结构：
     input -> DenseBlock1 -> TransitionLayer1 -> DenseBlock2 -> TransitionLayer2 -> .... -> Pooling -> Fc 
     
'''


# 稠密连接的卷积层
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate=0):
        """
        :param in_channels:输入通道数
        :param growth_rate: 卷积核的系数，也就是增长系数论文中的K
        :param bn_size: bottleneck层数的乘数，bn_size * k 瓶颈层的特征
        :param drop_rate: dropout的阈值
        """
        super(DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        # 构建卷积层
        self.dense_layer = nn.Sequential(
            # 批量归一化BN层
            nn.BatchNorm2d(in_channels),
            # 激活函数
            nn.ReLU(inplace=True),
            # 卷积
            nn.Conv2d(in_channels=in_channels, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            # 批量归一化BN层
            nn.BatchNorm2d(bn_size * growth_rate),
            # 激活函数
            nn.ReLU(inplace=True),
            # 卷积
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, padding=1, bias=False)
        )
        # dropout
        self.dropout = nn.Dropout(p=self.drop_rate)

    # 前向传播
    def forward(self, x):
        # 短连接
        identity = x
        # 稠密块
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        # 将二者结果连接起来，ResNet中是相加操作
        out = torch.cat([identity, y], dim=1)
        return out


# 过渡层
class TransitionLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        """
        由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型，过渡层用来控制模型复杂度。
        通过1x1卷积层来减小通道数，并使用stride=2的平均池化来进行下采样，减半图形的高与宽。（参考李沐《动手学深度学习》）
        :param in_channel:输入通道数
        :param out_channel:输出通道数
        """
        super(TransitionLayer, self).__init__()
        # 构建卷积
        self.transition_layer = nn.Sequential(
            # 批量归一化
            nn.BatchNorm2d(in_channel),
            # 激活函数
            nn.ReLU(inplace=True),
            # 卷积
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=False),
            # 平均池化，下采样
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    # 前向传播
    def forward(self, x):
        x = self.transition_layer(x)
        return x


# 稠密块
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channel, growth_rate, bn_size, drop_rate):
        """
        构建稠密连接网络的基础块结构
        :param num_layers:层数
        :param in_channel:输入通道数
        :param growth_rate:卷积核的系数，也就是增长系数论文中的K
        :param bn_size:bottleneck层数的乘数，bn_size * k 瓶颈层的特征
        :param drop_rate:dropout的系数
        """
        super(DenseBlock, self).__init__()
        # 卷积集合
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channel + i * growth_rate, growth_rate, bn_size, drop_rate))
        # 将卷积层封装成块
        self.block = nn.Sequential(*layers)

    # 前向传播
    def forward(self, x):
        return self.block(x)


# 构建DenseNet网络结构
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, num_init_features=64, block_config=[6, 12, 24, 16], bn_size=4, drop_rate=0,
                 num_classes=1000):
        """
        构建DenseNet网络
        :param growth_rate:卷积核的系数，也就是增长系数论文中的K
        :param num_init_features: 初始化的通道数
        :param block_config: 稠密块的初始配置
        :param bn_size:bottleneck层数的乘数，bn_size * k 瓶颈层的特征
        :param drop_rate:dropout的系数
        :param num_classes:分类数
        """
        super(DenseNet, self).__init__()
        # 开始的卷积部分，不涉及denseNet
        self.conv1 = nn.Sequential(
            # 卷积，输入：3x224x224   输出：num_init_features x112x112
            nn.Conv2d(in_channels=3, out_channels=num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            # 批量归一化
            nn.BatchNorm2d(num_init_features),
            # 激活函数
            nn.ReLU(inplace=True),
            # 最大池化,输入：num_init_features x112x112   输出：num_init_features x56x56
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 输入通道数
        num_features = num_init_features
        # 第一个稠密块结构
        self.dense1 = DenseBlock(num_layers=block_config[0], in_channel=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size, drop_rate=drop_rate)
        # 稠密块出来之后的通道数
        num_features = num_features + block_config[0] * growth_rate
        # 第一个过渡层，控制通道数 ，输出通道 //2
        self.transition1 = TransitionLayer(in_channel=num_features, out_channel=num_features // 2)
        # 经过过渡层之后的输出通道数减半
        num_features = num_features // 2
        # 第二个稠密块结构
        self.dense2 = DenseBlock(num_layers=block_config[1], in_channel=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size, drop_rate=drop_rate)
        # 第二个稠密块之后的通道数
        num_features = num_features + block_config[1] * growth_rate
        # 第二个过渡层,控制通道数，输出通道 //2
        self.transition2 = TransitionLayer(in_channel=num_features, out_channel=num_features // 2)
        # 经过过渡层之后的输出通道数减半
        num_features = num_features // 2
        # 第三个稠密块结构
        self.dense3 = DenseBlock(num_layers=block_config[2], in_channel=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size, drop_rate=drop_rate)
        # 稠密块出来之后的通道数
        num_features = num_features + block_config[2] * growth_rate
        # 第三个过渡层，控制通道数 ，输出通道 //2
        self.transition3 = TransitionLayer(in_channel=num_features, out_channel=num_features // 2)
        # 经过过渡层之后的输出通道数减半
        num_features = num_features // 2
        # 第四个稠密块结构
        self.dense4 = DenseBlock(num_layers=block_config[3], in_channel=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size, drop_rate=drop_rate)
        # 第四个稠密块之后的通道数
        num_features = num_features + block_config[3] * growth_rate
        # 平均池化，k=7，s=1
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        # 全连接
        self.fc = nn.Linear(num_features, num_classes)

    # 前向传播
    def forward(self, x):
        # 卷积部分
        x = self.conv1(x)
        # 第一个稠密块
        x = self.dense1(x)
        # 第一个过渡层
        x = self.transition1(x)
        # 第二个稠密块
        x = self.dense2(x)
        # 第二个过渡层
        x = self.transition2(x)
        # 第三个稠密块
        x = self.dense3(x)
        # 第三个过渡层
        x = self.transition3(x)
        # 第四个稠密块
        x = self.dense4(x)
        # 平均池化
        x = self.avg_pool(x)
        # 展开
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out


# DenseNet-121
def densenet121():
    return DenseNet(num_init_features=64, growth_rate=32, block_config=[6, 12, 24, 16])


# DenseNet-169
def densenet169():
    return DenseNet(num_init_features=64, growth_rate=32, block_config=[6, 12, 32, 32])


# DenseNet-201
def densenet201():
    return DenseNet(num_init_features=64, growth_rate=32, block_config=[6, 12, 48, 32])


# DenseNet-264
def densenet264():
    return DenseNet(num_init_features=64, growth_rate=32, block_config=[6, 12, 64, 48])


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = densenet264()
    print(model)
    model.to(device)
    X = torch.rand(2, 3, 224, 224)
    X = X.to(device)
    out = model(X)
    # g = make_dot(out)
    # g.view()
    print(out)
    # print(out.size())
    summary(model, (3, 224, 224))
