#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 21:35
# @Author  : zyf
# @File    : VGG_16.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary

'''
    经典CNN网络结构复现：LeNet、AlexNet、VGG、ResNet、InceptionNet等
    复现VGG16网络结构
    需要参考VGG16的网络结构，根据结构来构建模型，主要结构如下
    1.五个卷积部分
        第一部分包含两个卷积
        第二部分包含两个卷积
        第三部分包含三个卷积
        第四部分包含三个卷积
        第五部分包含三个卷积
        累计13层
    2.五个池化层
        接着上一个的五个卷积部分，每个部分都连接着一个池化层，降低图像的尺寸维度
        第一部分的卷积之后第一个池化层
        第二部分的卷积之后第二个池化层
        第三部分的卷积之后第三个池化层
        第四部分的卷积之后第四个池化层
        第五部分的卷积之后第五个池化层
    3.自定义池化层
        在所有的卷积层结束之后，跟着一个自定义池化层，目的是固定输出的维度大小
    3.三个全连接层
        第一个fc层，连接卷积的输出，input_features -> 4096 后有ReLu和Dropout
        第二个fc层，4096 -> 4096 后有ReLu和Dropout
        第三个fc层，4096 -> nums nums表示的分类数，vgg默认是1000类
'''


class VGG16(nn.Module):
    def __init__(self, nums):
        super(VGG16, self).__init__()
        self.nums = nums  # 分类数
        layers = []
        # 第一个卷积部分，包含两个卷积层和两个ReLu函数 64*224*224
        layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        # 第一个池化pooling,输出64*112*112
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # 第二个卷积部分，包含两个卷积层及两个ReLu函数,128*112*112
        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        # 第二个池化Pooling,输出128*56*56
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # 第三个卷积部分，包含三个卷积层及三个ReLu函数 256*56*56
        layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        # 第三个池化Pooling,输出256*28*28
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # 第四个卷积部分，包含三个卷积层及三个ReLu函数，512*28*28
        layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        # 第四个池化Pooling,输出512*14*14
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # 第五个卷积部分亦是最后一个卷积部分，同样包含三个卷积及三个ReLu函数，512*14*14
        layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        # 第五个池化亦是最后一个池化Pooling,输出512*7*7
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 将卷积层依次送入nn.Sequential,需要将list展开送入
        # 将每一个模块按照他们的顺序送入到nn.Sequential中,输入要么是orderdict,要么是一系列的模型，遇到上述的list，必须用*号进行转化
        self.features = nn.Sequential(*layers)
        print(layers)
        print(*layers)
        # 自适应池化Adaptive Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        # 全连接层实现方式，第一种
        fc = []
        # fc1,一个linear、relu、dropout
        fc.append(nn.Linear(in_features=512 * 7 * 7, out_features=4096))
        fc.append(nn.ReLU())
        fc.append(nn.Dropout())
        # fc2,一个linear、relu、dropout
        fc.append(nn.Linear(in_features=4096,out_features=4096))
        fc.append(nn.ReLU())
        fc.append(nn.Dropout())
        # fc3  一个linear
        fc.append(nn.Linear(in_features=4096,out_features=1000))
        self.classifer = nn.Sequential(*fc)

        # 第二种实现方式
        # self.classifier = nn.Sequential(
        #     nn.Linear(512*7*7,4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096,4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096,self.nums)
        # )
    def forward(self,x):
        print(x.size())
        print(x.size(0))
        x= self.features(x)
        x = self.avg_pool(x)
        #需要将多维度的值展平为一维，送入linear中，但是需要保持batchsize的维度
        # 例如2*512*7*7 变成2*25088
        # x= torch.flatten(x,1)
        x = x.view(x.size(0),-1)
        print(x)
        print(x.size())
        x = self.classifer(x)
        return x

# 测试数据
x = torch.rand((2,3,224,224))
vgg16 = VGG16(1000)
print(vgg16)
out = vgg16(x)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lenet = vgg16.to(device)
# 网络模型的数据流程及参数信息
summary(vgg16,(3,224,224))
