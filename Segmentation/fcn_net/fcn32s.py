#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/21 17:27
# @Author  : zyf
# @File    : fcn32s.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torchvision import models
from vgg import vgg
from torchsummary import summary

"""
    构建全卷积网络FCN,使用的backbone是VGG16

"""


class FCN32s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN32s, self).__init__()
        self.num_classes = num_classes
        backbone = vgg()
        # 对原图进行卷积conv1、pool1后图像缩小为1 / 2  featuremap
        self.layer1 = backbone.layer1
        self.pool1 = backbone.pool1
        # 对图像进行第二次卷积conv2、pool2后图像缩小为1 / 4 featuremap
        self.layer2 = backbone.layer2
        self.pool2 = backbone.pool2
        # 对图像进行第三次卷积conv3、pool3后图像缩小为1 / 8 featuremap
        self.layer3 = backbone.layer3
        self.pool3 = backbone.pool3
        # 对图像进行第四次卷积conv4、pool4后图像缩小为1 / 16 featuremap
        self.layer4 = backbone.layer4
        self.pool4 = backbone.pool4
        # ；对图像进行第五次卷积conv5、pool5后图像缩小为1 / 32 featuremap
        self.layer5 = backbone.layer5
        # 这里可以直接 进行类别的输出卷积了
        self.pool5 = backbone.pool5

        """
        所有的层都是卷积层，故称为全卷积网络。
        CNN操作过程中的全连接编程卷积操作的conv6、conv7，需要使用1*1的卷积
        图像的featuremap的大小依然为原图的1 / 32,此时图像不再叫featuremap而是叫heatmap.
        """
        # self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1)  # 512*7*7 -> 4096*7*7
        # self.drop6 = nn.Dropout()
        # self.fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)  # 4096*7*7 -> 4096*7*7
        # self.drop7 = nn.Dropout()

        # 输出结果    # 4096*7*7 -> num_classes*7*7
        self.score = nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=1)

        # 进行32倍上采样,使用双线性插值法，到达原始图像尺寸
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.pool1(x)  # feature map 1/2     224*224 -> 128*128
        x = self.layer2(x)
        x = self.pool2(x)  # feature map 1/4     224*224 -> 56*56
        x = self.layer3(x)
        x = self.pool3(x)  # feature map 1/8     224*224 -> 28*28
        x = self.layer4(x)
        x = self.pool4(x)  # feature map 1/16    224*224 -> 14*14
        x = self.layer5(x)
        x = self.pool5(x)  # feature map 1/32    224*224 -> 7*7

        # x = self.fc6(x)  # 512*7*7 -> 4096*7*7
        # x = self.drop6(x)
        # x = self.fc7(x)  # 4096*7*7 -> 4096*7*7
        # x = self.drop7(x)
        x = self.score(x)  # 4096*7*7 -> num_classes*7*7
        x = self.up_sample(x)  # num_classes*7*7 ->  num_classes*(7x32)*(7x32)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FCN32s()
    net = net.to(device)
    print(net)
    data = torch.rand((2, 3, 224, 224))
    result = net(data.to(device))
    print(result.size())
    summary(net, (3, 224, 224))