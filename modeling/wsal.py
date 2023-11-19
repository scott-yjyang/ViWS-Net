import math
from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models
from modeling.decoder import ConvLayer, UpsampleConvLayer, ResidualBlock
from modeling.functions import ReverseLayerF
import torch
import math



class AttentionModule(nn.Module):
    """
    The attention module takes multiple hidden representations and compute the attention-weighted average
    Use Gated Attention Mechanism in https://arxiv.org/pdf/1802.04712.pdf
    """
    def __init__(self, dim=512, num_types=3):
        """
        Function that add layers to the parent module that implements nn.Module
        :return:
        """
        super(AttentionModule, self).__init__()
        # The gated attention mechanism
        self.mil_attn_V = nn.Linear(dim, 128, bias=False)
        self.mil_attn_U = nn.Linear(dim, 128, bias=False)
        self.mil_attn_w = nn.Linear(128, 1, bias=False)
        # classifier
        #self.classifier_linear = nn.Sequential(
        #    nn.Linear(dim, dim),
            #nn.BatchNorm1d(512),
        #    nn.ReLU(),
            #nn.Dropout(0.1),
        #    nn.Linear(dim, num_types),
        #)
        self.classifier_linear = nn.Linear(dim, num_types, bias=False)

    def forward(self, x):
        """
        Function that takes in the hidden representations of crops and use attention to generate a single hidden vector
        :param h_small:
        :param h_crops:
        :return:
        """
        batch_size, num_crops, h_dim = x.size()
        x_reshape = x.view(batch_size * num_crops, h_dim)
        # calculate the attn score
        attn_projection = torch.sigmoid(self.mil_attn_U(x_reshape)) * \
                          torch.tanh(self.mil_attn_V(x_reshape))
        attn_score = self.mil_attn_w(attn_projection)
        # use softmax to map score to attention
        attn_score_reshape = attn_score.view(batch_size, num_crops)
        attn = F.softmax(attn_score_reshape, dim=1)

        # final hidden vector
        z_weighted_avg = torch.sum(attn.unsqueeze(-1) * x, 1)

        # map to the final layer
        y = self.classifier_linear(z_weighted_avg)
        return  y

class WSAL(nn.Module):
    def __init__(self, dim=512, num_types=3):
        super(WSAL, self).__init__()
        self.num_types = num_types
        #self.q = nn.Linear(256, 256, bias=False)
        #self.decouple = nn.Sequential(
        #    ResBlock(in_feat=64, out_feat=128, stride=2),
        #    ResBlock(in_feat=128, out_feat=256, stride=2),
        #    #ResBlock(in_feat=256, out_feat=512, stride=2),
        #    #nn.AdaptiveAvgPool2d(1)
        #)
        #self.conv = ConvLayer(512,512,3,1,1)
        self.feature_avgpool = nn.AdaptiveAvgPool2d(1)
        #self.conv = ResidualBlock(512)
        #self.weather_pred = nn.Sequential(
        #    nn.Linear(512*5, 512),
            #nn.BatchNorm1d(512),
        #    nn.ReLU(),
            #nn.Dropout(0.1),
        #    nn.Linear(512, num_types),
        #)
        self.weather_pred = AttentionModule(dim,num_types)
        #self.cl_proj = nn.Sequential(
        #    nn.Linear(256, 256),
        #    nn.LeakyReLU(0.1, True),
        #    nn.Linear(256, 256),
        #                )
        #self.cl_criterion = InfoNCE()
    
    def forward(self, x, alpha, B, T):
        ## weather-aware
        #print(x.shape)
        #x_ = x.view(-1,512*4*4)
        reverse_x = ReverseLayerF.apply(x, alpha)
        #reverse_x = self.conv(reverse_x)
        x_ = self.feature_avgpool(reverse_x)
        x_ = torch.flatten(x_, 1)
        #x_ = x_.view(B, T, -1).view(B,-1)
        x_ = x_.view(B, T, -1)
        #x_ = torch.cat([x_,x_[:,T//2:T//2+1]],dim=1)
        #fused_x = torch.sum(x_, dim=1)

        #print(fused_x.shape)
        pred = self.weather_pred(x_)


        return pred, reverse_x
