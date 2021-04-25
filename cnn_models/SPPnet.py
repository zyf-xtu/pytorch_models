#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 10:12
# @Author  : zyf
# @File    : SPPnet.py
# @Software: PyCharm
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

'''
    空间金字塔池化（spatial pyramid pooling）
    SPPnet的网络结构能够产生固定大小的表示（representation）不管输入图像的尺寸和比例。
    主要是解决输入图像需要固定尺寸的问题。当前深度卷积神经网络CNNS都需要输入的图像尺寸固定，
    这种人为的需要导致面对任意尺寸和比例的图像或者子图像时降低识别的精度（因为需要经过crop/warp变换)。
    那么为什么CNNs需要一个固定的输入尺寸呢？CNN主要由两部分组成，卷积部分和其后的全连接部分。
    卷积部分通过滑窗进行计算，并输出代表激活的空间排列的特征图（feature map）。事实上，卷积并不需要固定的图像尺寸，
    他可以产生任意尺寸的特征图。而另一方面，根据定义，全连接层则需要固定的尺寸输入。因此固定尺寸的问题来源于全连接层，也是网络的最后阶段。
    空间金字塔池化( spatial pyramid pooling，SPP)层以移除对网络固定尺寸的限制。特别地，将SPP层放在最后一个卷积层之后。
    SPP层对特征图进行池化，并产生固定长度的输出，这个输出再喂给全连接层（或其他分类器）。
    换句话说，在网络层次的较后阶段（也就是卷积层和全连接层之间）进行某种信息“汇总”，
    可以避免在最开始的时候就进行裁剪crop或变形warp。网络结构为SPP-net。
    传统的CNN流程：
    image -> crop/warp(固定尺寸) -> conv layers -> fc layers -> output
    加入SPP的CNN流程：
    image -> conv layers -> spatial pyramid pooling -> fc layers -> output
    计算公式：
        向下取整：floor()
        向上取整：ceil()
        池化后矩阵大小计算公式：
        没有步长：(h+2p-f+1)*(w+2p-f+1)
        有步长：ceil((h+2p-f)/s +1)*ceil((h+2p-f)/s +1)
   求spp层的kernel_size、padding、stride的值
        已知条件：h_in,w_in 表示输入的高宽，n表示池化数量
            kernel_size = (ceil(h_in/n),ceil(w_in/n) 
            stride = (ceil(h_in/n),ceil(w_in/n)
            padding = (floor((kernel_size[0]*n-h_in+1)/2),floor((kernel_size[1]*n-w_in+1)/2))

    参考：https://www.cnblogs.com/marsggbo/p/8572846.html
'''


# 构建SPP层(空间金字塔池化)
class SPPLayer(nn.Module):
    def __init__(self, num_pools=[1, 2, 4], pool_type='max_pool'):
        """
        在卷积层后面实现空间金字塔池化，固定输出的尺寸送入到全连接层
        :param num_pools: 论文中的[1,2,4],表示需要将原先的特征图分别划分1x1、2x2、4x4三种
        :param pool_type: 池化类型，最大池化和平均池化
        """
        super(SPPLayer, self).__init__()
        self.num_pools = num_pools
        self.pool_type = pool_type

    # 前向传播
    def forward(self, x):
        num, channel, width, height = x.size()  # 批数量、通道数、高、宽
        print(x.size())
        for i in range(len(self.num_pools)):
            # 计算卷积核的尺寸,ceil向上取整
            kernel_size = (math.ceil(height / self.num_pools[i]), math.ceil(width / self.num_pools[i]))
            # 计算步长的大小，向上取整
            stride_size = (math.ceil(height / self.num_pools[i]), math.ceil(width / self.num_pools[i]))
            # 计算padding的大小，向下取整
            padding = (math.floor((kernel_size[0] * self.num_pools[i] - height + 1) / 2),
                       math.floor((kernel_size[1] * self.num_pools[i] - width + 1) / 2))
            print('i%s   kernel %s   stride %s   padding %s'%(i,kernel_size,stride_size,padding))
            # 选择池化方式
            if self.pool_type == 'max_pool':
                pool = F.max_pool2d(x, kernel_size=kernel_size, stride=stride_size, padding=padding)
            else:
                pool = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride_size, padding=padding)
            print(pool.size())
            # 将池化后的结果展开，拼接
            if i == 0:
                x_flatten = pool.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, pool.view(num, -1)), -1)
        print(x_flatten.size())

        return x_flatten


x = torch.rand((32, 256, 13, 13))
model = SPPLayer()
print(model)
out = model(x)
print(out)
