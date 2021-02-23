#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 10:36
# @Author  : zyf
# @File    : VGG_19.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchsummary import summary
# from torchviz import make_dot

'''
    手撕VGG19网络结构
    需要参考VGG19的网络结构，根据结构来构建模型，主要结构如下
    1.五个卷积部分
        第一部分包含两个卷积
        第二部分包含两个卷积
        第三部分包含四个卷积
        第四部分包含四个卷积
        第五部分包含四个卷积
        累计16层
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


# VGG自定义类
class VGG19(nn.Module):
    def __init__(self, nums):  # nums 是要分类的类别数
        super(VGG19, self).__init__()
        self.nums = nums
        # 卷积层
        self.features = nn.Sequential(
            # 第一个卷积部分包含两个卷积层，每个卷积后都有一个ReLu函数,
            # 第一个卷积层输入输入：batch_size*3*224*224,输出：batch_size*64*224*224
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第一个池化层，输入：batch_size*64*224*224, 输出：batch_size*64*112*112
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第二个卷积部分，包含两个卷积层。第一个卷积层输入输入是：batch_size*64*112*112 ,输出是batch_size*128*112*112
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第二个池化层，输入是batch_size*128*112*112,输出是:batch_size*128*56*56
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第三个卷积部分，包含四个卷积层。第一个卷积层输入是batch_size*128*56*56, 输出是：batch_size*256*56*56
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第三个池化，输入：batch_size*256*56*56 输出：batch_size*256*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第四个卷积部分，包含四个卷积层。第一个卷积层的输入是：batch_size*256*28*28  输出是：batch_size*512*28*28
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第四个池化层，输入：batch_size*512*28*28  输出batch_size*512*14*14
            nn.MaxPool2d(kernel_size=2, stride=1),
            # 第五个卷积部分，包含四个卷积层。第一个卷积层的输入：batch_size*512*14*14 输出：batch_size*512*14*14
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第五个池化，也是最后一个池化，输入：batch_size*512*14*14  输出：batch_size*512*7*7
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 自定义池化层，目的是固定输出的维度大小(7*7)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        # 全连接层
        self.classifer = nn.Sequential(
            # 第一个全连接层fc1,输入：batch_size*512*7*7  输出：4096
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            # 第二个全连接层fc2,输入：4096  输出：4096
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            # 第三个全连接层fc3,输入：4096  输出：nums 即分类数
            nn.Linear(in_features=4096, out_features=self.nums)
        )

    def forward(self, x):
        # 首先进入卷积层
        x = self.features(x)
        # 自定义池化层
        x = self.avg_pool(x)
        # 需要将多维度的值展平为一维，送入linear中，但是需要保持batchsize的维度
        # 例如2*512*7*7 变成2*25088
        x = x.view(x.size(0), -1)
        # 调用全连接层
        x = self.classifer(x)
        return x


# 测试数据
x = torch.rand((8, 3, 224, 224))
vgg19 = VGG19(1000)
print(vgg19)
out = vgg19(x)
print(out)
# g = make_dot(out)
# g.view()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg19 = vgg19.to(device)
# 展示网络模型的数据流向及参数信息
summary(vgg19,(3,224,224))
