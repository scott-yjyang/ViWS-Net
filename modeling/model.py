import math
from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models


from timm.models.vision_transformer import *
from timm.models import create_model
from modeling.backbone import *
from modeling.decoder import *
from modeling.transweather import Tdec
from modeling.wsal import *


class RefineNet(nn.Module):

    def __init__(self, **kwargs):
        super(RefineNet, self).__init__()

        self.backbone = create_model('shunted_t',
            pretrained=False,
            num_classes=4,
            drop_rate=0,
            drop_path_rate=0.3,
            drop_block_rate=None,
        )
        
        self.convproj = convprojection_base()

        self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()

        
        self.init_weight(path='./models/ckpt_T.pth')

    def init_weight(self,path=None):
        
        if path:
            if path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(path)

            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = self.backbone.state_dict()
            # # 1. filter out unnecessary keys
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if 'head' not in k}
            #print(checkpoint_model.keys())
            # 2. overwrite entries in the existing state dict
            state_dict.update(checkpoint_model)
            
            self.backbone.load_state_dict(checkpoint_model, strict=False)
    
    def forward(self, x, B, T):

        base = x

        x1,_ = self.backbone(x, B, T)

        x = self.convproj(x1)

        clean = self.active(self.clean(x)) + base

        return clean


class ViWSNet(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """

    def __init__(self, config):
        super(ViWSNet, self).__init__()

        self.config = config

        self.backbone = create_model('shunted_s',
            pretrained=False,
            num_classes=4,
            drop_rate=0,
            drop_path_rate=0.3,
            drop_block_rate=None,
        )
        self.init_weight()
        self.weather_decoder = Tdec()

        ## weather aware
        self.weather_aware = WSAL(dim=512,num_types=self.config['num_types'])

        self.convtail = convprojection()
        self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()
        
        self.tail = nn.Sequential(
                    MyConv3d(in_channels=3,
                             out_channels=32,
                             kernel_size=3,
                             stride=1,
                             bias=False),
                    nn.ReLU(True),
                    MyConv3d(in_channels=32,
                             out_channels=32,
                             kernel_size=3,
                             stride=1,
                             bias=False),
                    nn.ReLU(True),
                    MyConv3d(in_channels=32,
                             out_channels=3,
                             kernel_size=3,
                             stride=1,
                             bias=False)
                )
        
        self.refine = RefineNet()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
        
        if self.config['finetune']:
            if self.config['finetune'].startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    self.config['finetune'], map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(self.config['finetune'])

            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = self.backbone.state_dict()
            # # 1. filter out unnecessary keys
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if 'head' not in k}
            #print(checkpoint_model.keys())
            # 2. overwrite entries in the existing state dict
            state_dict.update(checkpoint_model)
            
            self.backbone.load_state_dict(checkpoint_model, strict=False)


    def forward_adv(self, x, B, T, alpha=None):
        #print(imgs.shape)
        bt, num_frames, c, h, w = x.shape
        base = x.transpose(1, 2)
        #print(base.shape)
        frames = x.reshape(bt*num_frames,c,h,w)
        
        x1, msg_token = self.backbone(frames,B,T)
        weather_pred, x_adv = self.weather_aware(x1[-1],alpha, bt, num_frames)
        x2 = self.weather_decoder(x1[-1], msg_token[-1])
        #weather_pred, features = self.weather_aware(x2[0],dc,B,T)

        x, x_last = self.convtail(x1,x2)
        x = self.active(self.clean(x))
        x = x.reshape(bt,num_frames,x.shape[1],x.shape[2],x.shape[3]).transpose(1, 2)
        clean = (self.tail(base - x)).transpose(1, 2).reshape(bt*num_frames,c,h,w)
        final_out = self.refine(clean, bt, num_frames)
        #return final_out
        return final_out, weather_pred

    def inference(self, x, B, T, dc=None):
        #print(imgs.shape)
        bt, num_frames, c, h, w = x.shape
        base = x.transpose(1, 2)
        #print(base.shape)
        frames = x.reshape(bt*num_frames,c,h,w)
        
        x1, msg_token = self.backbone(frames,B,T)
        #features, x_adv = self.weather_aware(x1[-1], 1.0, bt, num_frames)
        x2 = self.weather_decoder(x1[-1], msg_token[-1])
        #weather_pred, features = self.weather_aware(x2[0],dc,B,T)

        x, x_last = self.convtail(x1,x2)
        #features = self.weather_aware(x_last,dc,B,T)
        x = self.active(self.clean(x))
        x = x.reshape(bt,num_frames,x.shape[1],x.shape[2],x.shape[3]).transpose(1, 2)
        clean = (self.tail(base - x)).transpose(1, 2).reshape(bt*num_frames,c,h,w)
        final_out = self.refine(clean, bt, num_frames)
        return final_out

    def forward(self, x, B, T, alpha=None, phase='train'):
        if phase == 'train':
            return self.forward_adv(x, B, T, alpha)
        elif phase == 'test':
            return self.inference(x, B, T, alpha)


