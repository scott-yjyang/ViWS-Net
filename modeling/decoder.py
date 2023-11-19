

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
#         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        #self.upconv2 = nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1)
        #self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out
    


class MyConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super(MyConv3d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(0, int((kernel_size-1)/2), int((kernel_size-1)/2)),
                              bias=bias)

    def forward(self, x):
        x = F.pad(x, pad=(0,)*4+(int((self.kernel_size-1)/2),)*2, mode='replicate')
        return self.conv(x)



class convprojection(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection,self).__init__()

        self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 256, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(256))
        self.convd8x = UpsampleConvLayer(256, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential( ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)
        #self.conv_output = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        #self.active = nn.Tanh()        

    def forward(self,x1,x2):

        res32x = self.convd32x(x2[0])

        if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0,-1,0,-1)
            res32x = F.pad(res32x,p2d,"constant",0)
            
        elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
            p2d = (0,-1,0,0)
            res32x = F.pad(res32x,p2d,"constant",0)
        elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0,0,0,-1)
            res32x = F.pad(res32x,p2d,"constant",0)

        res16x = res32x + x1[3]
        res16x = self.convd16x(res16x) 

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,-1,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0,-1,0,0)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,0,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)

        res8x = self.dense_4(res16x) + x1[2]
        x_last = res8x
        res8x = self.convd8x(res8x) 
        
        if x1[1].shape[3] != res8x.shape[3] and x1[1].shape[2] != res8x.shape[2]:
            p2d = (0,-1,0,-1)
            res8x = F.pad(res8x,p2d,"constant",0)
        elif x1[1].shape[3] != res8x.shape[3] and x1[1].shape[2] == res8x.shape[2]:
            p2d = (0,-1,0,0)
            res8x = F.pad(res8x,p2d,"constant",0)
        elif x1[1].shape[3] == res8x.shape[3] and x1[1].shape[2] != res8x.shape[2]:
            p2d = (0,0,0,-1)
            res8x = F.pad(res8x,p2d,"constant",0)

        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)

        if x1[0].shape[3] != res4x.shape[3] and x1[0].shape[2] != res4x.shape[2]:
            p2d = (0,-1,0,-1)
            res4x = F.pad(res4x,p2d,"constant",0)
        elif x1[0].shape[3] != res4x.shape[3] and x1[0].shape[2] == res4x.shape[2]:
            p2d = (0,-1,0,0)
            res4x = F.pad(res4x,p2d,"constant",0)
        elif x1[0].shape[3] == res4x.shape[3] and x1[0].shape[2] != res4x.shape[2]:
            p2d = (0,0,0,-1)
            res4x = F.pad(res4x,p2d,"constant",0)

        res2x = self.dense_2(res4x) + x1[0]
        
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)

        return x, x_last

class convprojection_base(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection_base,self).__init__()

        # self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 256, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(256))
        self.convd8x = UpsampleConvLayer(256, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential( ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)
        self.conv_output = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()        

    def forward(self,x1):

#         if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
#             p2d = (0,-1,0,-1)
#             res32x = F.pad(res32x,p2d,"constant",0)
            
#         elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
#             p2d = (0,-1,0,0)
#             res32x = F.pad(res32x,p2d,"constant",0)
#         elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
#             p2d = (0,0,0,-1)
#             res32x = F.pad(res32x,p2d,"constant",0)

#         res16x = res32x + x1[3]
        res16x = self.convd16x(x1[3]) 

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,-1,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0,-1,0,0)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,0,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)

        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x) 

        if x1[1].shape[3] != res8x.shape[3] and x1[1].shape[2] != res8x.shape[2]:
            p2d = (0,-1,0,-1)
            res8x = F.pad(res8x,p2d,"constant",0)
        elif x1[1].shape[3] != res8x.shape[3] and x1[1].shape[2] == res8x.shape[2]:
            p2d = (0,-1,0,0)
            res8x = F.pad(res8x,p2d,"constant",0)
        elif x1[1].shape[3] == res8x.shape[3] and x1[1].shape[2] != res8x.shape[2]:
            p2d = (0,0,0,-1)
            res8x = F.pad(res8x,p2d,"constant",0)

        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)

        if x1[0].shape[3] != res4x.shape[3] and x1[0].shape[2] != res4x.shape[2]:
            p2d = (0,-1,0,-1)
            res4x = F.pad(res4x,p2d,"constant",0)
        elif x1[0].shape[3] != res4x.shape[3] and x1[0].shape[2] == res4x.shape[2]:
            p2d = (0,-1,0,0)
            res4x = F.pad(res4x,p2d,"constant",0)
        elif x1[0].shape[3] == res4x.shape[3] and x1[0].shape[2] != res4x.shape[2]:
            p2d = (0,0,0,-1)
            res4x = F.pad(res4x,p2d,"constant",0)

        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)

        return x
