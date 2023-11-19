import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import random


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.dropout_layer = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, identity=None, msg_tokens=None):
        out = self.fc1(x)
        out = self.act(out + self.dwconv(out, H, W))
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)

        bt, msg_nums, d = msg_tokens.size()

        msg_tokens = torch.einsum('bnc,oc->bno', [msg_tokens, self.fc1.weight]) + self.fc1.bias
        msg_tokens = msg_tokens + torch.einsum('bnc,c->bnc', [msg_tokens, self.dwconv.dwconv.weight.sum((-1, -2, -3))]) + self.dwconv.dwconv.bias
        msg_tokens = self.drop(self.act(msg_tokens))
        msg_tokens = torch.einsum('bnc,oc->bno', [msg_tokens, self.fc2.weight]) + self.fc2.bias
        msg_tokens = self.drop(msg_tokens)

        if identity is None:
            identity = x
        out = torch.cat([out, msg_tokens], dim=1)
        out = identity + self.dropout_layer(out)

        return out[:, :-msg_nums, ...], out[:, -msg_nums:, ...]


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, drop_path=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dropout_layer = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio==8:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==4:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==2:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
            self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, identity=None, msg_tokens=None):
        bt, msg_nums, d = msg_tokens.size()
        x_q = torch.cat([x, msg_tokens], dim=1)
        B, N, C = x_q.shape
        q = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                
                x_1 = self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)
                msg_tokens1 = torch.einsum('bnc,oc->bno', [msg_tokens, self.sr1.weight.sum((-1, -2))]) + self.sr1.bias
                x_1 = torch.cat([x_1, msg_tokens1], dim=1)
                x_1 = self.act(self.norm1(x_1))

                x_2 = self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)
                msg_tokens2 = torch.einsum('bnc,oc->bno', [msg_tokens, self.sr2.weight.sum((-1, -2))]) + self.sr2.bias
                x_2 = torch.cat([x_2, msg_tokens2], dim=1)
                x_2 = self.act(self.norm2(x_2))   

                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)

                k1, v1 = kv1[0], kv1[1] #B head N C
                k2, v2 = kv2[0], kv2[1]

                attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                #print(v1.transpose(1, 2).reshape(B, -1, C//2).transpose(1, 2).shape)
                tmp_msg1 = v1.transpose(1, 2).reshape(B, -1, C//2).transpose(1, 2)[:,:,-msg_nums:]
                #print(tmp_msg1.shape)
                v1 = v1 + torch.cat([self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2)[:,:,:-msg_nums].view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
                    view(B, C//2, -1),tmp_msg1],2).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)
                attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                #print(v2.transpose(1, 2).reshape(B, -1, C//2).transpose(1, 2).shape)
                tmp_msg2 = v2.transpose(1, 2).reshape(B, -1, C//2).transpose(1, 2)[:,:,-msg_nums:]
                v2 = v2 + torch.cat([self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2)[:,:,:-msg_nums].view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
                    view(B, C//2, -1),tmp_msg2],2).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)

                x = torch.cat([x1,x2], dim=-1)
        else:
            x = torch.cat([x, msg_tokens],dim=1)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            #print(attn.shape)
            tmp_msg = v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2)[:,:,-msg_nums:]
            #print(v.transpose(1, 2).reshape(B, -1, C).shape)
            x = (attn @ v).transpose(1, 2).reshape(B, -1, C) + torch.cat([self.local_conv(v.transpose(1, 2).reshape(B, -1, C).
                                        transpose(1, 2)[:,:,:-msg_nums].view(B,C, H, W)).view(B, C, -1),tmp_msg],2).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        if identity is None:
            identity = x_q
        

        out = identity + self.dropout_layer(x)
        return out[:, :-msg_nums, ...], out[:, -msg_nums:, ...]


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, drop_path=drop_path)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, drop_path=drop_path)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, B, T, msg_tokens, msg_shift=None):
        #print(x.shape)
        #print(msg_tokens.shape)
        x, msg_tokens = self.attn(self.norm1(x), H, W, 
                            msg_tokens = self.norm1(msg_tokens), identity=torch.cat([x, msg_tokens], dim=1))
        x, msg_tokens = self.mlp(self.norm2(x), H, W,
                            msg_tokens = self.norm2(msg_tokens), identity=torch.cat([x, msg_tokens], dim=1))

        if msg_shift is not None:
            msg_tokens = msg_tokens.reshape(B, T, *msg_tokens.size()[1:])
            msg_tokens = msg_tokens.chunk(len(msg_shift), dim=2)
            msg_tokens = [torch.roll(tokens, roll, dims=1) for tokens, roll in zip(msg_tokens, msg_shift)]
            msg_tokens = torch.cat(msg_tokens, dim=2).flatten(0, 1)

        return x, msg_tokens


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, msg_tokens):
        x = self.proj(x)
        msg_tokens = torch.einsum('bnc,oc->bno', [msg_tokens, self.proj.weight.sum((-1, -2))]) + self.proj.bias

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        msg_tokens = self.norm(msg_tokens)
        return x, H, W, msg_tokens

class Head(nn.Module):
    def __init__(self, num):
        super(Head, self).__init__()
        stem = [nn.Conv2d(3, 64, 7, 2, padding=3), nn.BatchNorm2d(64), nn.ReLU(True)]
        for i in range(num):
            stem.append(nn.Conv2d(64, 64, 3, 1, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(64))
            stem.append(nn.ReLU(True))
        stem.append(nn.Conv2d(64, 64, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(64)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, msg_tokens):
        x = self.conv(x)
        '''
        for i,layer in enumerate(self.conv[:-1]):
            if isinstance(layer, nn.Conv2d): 
                print(layer)
                msg_tokens = torch.einsum('bnc,oc->bno', [msg_tokens, self.conv[i].weight.sum((-1, -2))])     
            else: 
                print(layer)
                msg_tokens = self.conv[i](msg_tokens)
        '''
        #print(msg_tokens.shape)
        msg_tokens = torch.einsum('bnc,oc->bno', [msg_tokens, self.conv[0].weight.sum((-1, -2))]) + self.conv[0].bias
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        msg_tokens = self.norm(msg_tokens)
        return x, H,W, msg_tokens

class ShuntedTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, num_conv=0, out_indices=(0, 1, 2, 3), num_msg_tokens=48, shift_strides=[1,0,-1,2,0,-2]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.num_msg_tokens = num_msg_tokens
        self.shift_strides = shift_strides
        self.msg_tokens = nn.Parameter(
            torch.zeros(1, num_msg_tokens, in_chans)).requires_grad_(True)
        trunc_normal_(self.msg_tokens, std=.02)
        self.msg_shift = []

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            shifts = shift_strides
            self.msg_shift.append([])
            if i ==0:
                patch_embed = Head(num_conv)#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            for lid in range(depths[i]):
                if lid % 2 == 0:
                    self.msg_shift[-1].append([_ for _ in shifts])
                else:
                    self.msg_shift[-1].append([-_ for _ in shifts])
                    #if lid != depths[i]-1 and (lid+1)%4==0:
                    #    perm1 = torch.arange(int(len(shifts)/2)-1, -1, -1)
                    #    perm2 = torch.arange(len(shifts)-1,int(len(shifts)/2)-1,-1)
                    #    perm = torch.cat([perm2,perm1],0)
                    #    shifts = [shifts[i] for i in perm]
                    #    perm = torch.randperm(len(shifts))
                    #    shifts = [shifts[i] for i in perm]
                        #print(shifts)
                        #print(shifts)
                        

            if depths[i] % 2 == 1:
                self.msg_shift[-1][-1] = None
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            #print(self.msg_shift)
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        #self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, B, T):

        outs = []
        tokens = []
        msg_tokens = self.msg_tokens.repeat(B * T, 1, 1)
        
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W, msg_tokens = patch_embed(x, msg_tokens)
            #print(x.shape)
            for _, blk in enumerate(block):
                x, msg_tokens = blk(x, H, W, B, T, msg_tokens, msg_shift=self.msg_shift[i][_])
            #x = norm(x)
            x, msg_tokens = norm(x), norm(msg_tokens)
            x = x.reshape(B*T, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(x)
                tokens.append(msg_tokens)


        return outs, tokens
    '''
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x
    '''

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict



@register_model
def shunted_t(pretrained=False, **kwargs):
    model = ShuntedTransformer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 2, 4, 1], sr_ratios=[8, 4, 2, 1], num_conv=0,
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def shunted_s(pretrained=False, **kwargs):
    model = ShuntedTransformer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 12, 1], sr_ratios=[8, 4, 2, 1], num_conv=1, **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def shunted_b(pretrained=False, **kwargs):
    model = ShuntedTransformer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 24, 2], sr_ratios=[8, 4, 2, 1], num_conv=2,
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def shunted_weather(pretrained=False, **kwargs):
    model = ShuntedTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 8, 8], mlp_ratios=[2, 2, 2, 2], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], num_conv=2,
        **kwargs)
    model.default_cfg = _cfg()

    return model