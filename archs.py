import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from thop import profile,clever_format
import os
from utils import *

import time
import pywt
import pywt.data
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
__all__ = ['CTM-UNet']

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from timm.models.layers import DropPath, trunc_normal_
import types
import math


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


## ShiftedMLP
class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(in_features, hidden_features)
        self.fc4 = nn.Linear(in_features, hidden_features)
        self.fc5 = nn.Linear(in_features * 2, hidden_features)
        self.fc6 = nn.Linear(in_features * 2, out_features)
        self.drop = nn.Dropout(drop)
        self.dwconv = DWConv(hidden_features)
        self.act1 = act_layer()
        self.act2 = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_features * 2)
        self.norm2 = nn.BatchNorm2d(hidden_features)

        self.shift_size = shift_size
        self.pad = shift_size // 2
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

    def forward(self, x, H, W):
        B, N, C = x.shape

        ### shifted-MLP
        ### 第一次shifted-MLP 宽度
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc1(x_shift_r)
        x_shift_r = self.act1(x_shift_r)
        x_shift_r = self.drop(x_shift_r)

        ### 第二次shifted-MLP 高度
        xn = x_shift_r.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc2(x_shift_c)
        x_1 = self.drop(x_shift_c)

        ### 第三次shifted-MLP 高度反方向
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, -shift, 3) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc3(x_shift_c)
        x_shift_c = self.act1(x_shift_c)
        x_shift_c = self.drop(x_shift_c)

        ### 第四次次shifted-MLP 宽度
        xn = x_shift_c.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc4(x_shift_r)
        x_2 = self.drop(x_shift_r)

        # 进行特征融合和输出处理
        x_1 = torch.add(x_1, x)
        x_2 = torch.add(x_2, x)
        x1 = torch.cat([x_1, x_2], dim=2)
        x1 = self.norm1(x1)
        x1 = self.fc5(x1)
        x1 = self.drop(x1)
        x1 = torch.add(x1, x)
        x2 = x.transpose(1, 2).view(B, C, H, W)

        ### DSC
        x2 = self.dwconv(x2, H, W)
        x2 = self.act2(x2)
        x2 = self.norm2(x2)
        x2 = x2.flatten(2).transpose(1, 2)

        out = torch.cat([x1, x2], dim=2)
        out = self.fc6(out)
        out = self.drop(out)
        return out

class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
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
    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = GlobalFilter(dim, dim, 3, 1, bias=True)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x, H, W):
        x = self.dwconv(x)
        x = self.point_conv(x)
        return x



# DSE

class DSE(nn.Module):
    def __init__(self, channel, reduction=2):
        super(DSE, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x)
        y1 = self.layer1(y1)
        x = x * y1
        y2 = self.max_pool(x)
        y2 = self.layer2(y2)

        return x * y2.expand_as(x)


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
def wavelet_transform_init(filters):
    class WaveletTransform(Function):

        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = inverse_wavelet_transform(grad_output, filters)
            return grad, None

    return WaveletTransform().apply
def inverse_wavelet_transform_init(filters):
    class InverseWaveletTransform(Function):

        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = inverse_wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = wavelet_transform(grad_output, filters)
            return grad, None

    return InverseWaveletTransform().apply
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

# WTConv全局滤波器
class GlobalFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(GlobalFilter, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class MSA(nn.Module):
    def __init__(self, dim, scales=1):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        kernel_sizes = [1, 3, 5, 7]
        paddings = [0, 1, 2, 3]
        for i in range(self.nums) :
            kernel_size = kernel_sizes[i % len(kernel_sizes)]
            padding = paddings[i % len(paddings)]
            convs.append(nn.Conv2d(width,width,kernel_size=kernel_size,padding=padding,groups=width))

            # convs.append(nn.Conv2d(width, width, kernel_size=1,padding=0, groups=width))
            # convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width))
            # convs.append(nn.Conv2d(width, width, kernel_size=5, padding=2, groups=width))
            # convs.append(nn.Conv2d(width, width, kernel_size=7, padding=3, groups=width))
        self.convs = nn.ModuleList(convs)

        self.att = SE(dim)
    def forward(self, x):
        x_ = x
        spx = torch.split(x, self.width, 1)
        t = spx[0]
        out = t
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = t + spx[i]
            sp = self.convs[i](sp)
            t = t + sp
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)
        x = self.att(x)
        x = torch.add(x_, x)
        return x



#CSCA
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x
class csa(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim,eps=1e-6,data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim,dim,1),
            nn.GELU(),
            nn.Conv2d(dim,dim,7,padding=3,groups=dim)# 11x11,5  7x7,3已跑：91.7   5x5,2  3x3,1  9x9,4 21x21，10
        )
        self.v = nn.Conv2d(dim,dim,1)
        self.proj = nn.Conv2d(dim, dim,1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x
class CSA(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.):
        super().__init__()

        self.attn = csa(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


def calc_uncertainty(score):

    # seg shape: bs, obj_n, h, w
    # print("xxxxxxxxxxxxxxxxxx")
    # print(score.shape)
    score_top = score.topk(k=1, dim=1)
    # print(score_top[0].shape)
    # print("zzzzzz")
    # print(type(score_top[:, 0]))
    uncertainty = score_top[0] / (score + 1e-8)  # bs, h, w   M1/M2
    uncertainty = torch.exp(1 - uncertainty)  # bs, 1, h, w   不确定映射 U = exp(1-M1/M2)
    return uncertainty  # 返回 不确定映射 U
class CTM-UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=256,
                 num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1],**kwargs):
        super().__init__()

        # self.filters = [8, 16, 32, 64, 128]    # S
        # self.filters = [16, 32, 128, 160, 256]    # M
        self.filters = [32, 64, 128, 256, 512]    # L

        self.encoder1 = nn.Conv2d(input_channels, self.filters[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(self.filters[0], self.filters[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(self.filters[1], self.filters[2], 3, stride=1, padding=1)
        self.encoder4 = nn.Conv2d(self.filters[2], self.filters[3], 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(self.filters[3], self.filters[4], 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(self.filters[0])
        self.ebn2 = nn.BatchNorm2d(self.filters[1])
        self.ebn3 = nn.BatchNorm2d(self.filters[2])
        self.ebn4 = nn.BatchNorm2d(self.filters[3])
        self.ebn5 = nn.BatchNorm2d(self.filters[4])

        self.norm1 = norm_layer(self.filters[0])
        self.norm2 = norm_layer(self.filters[1])
        self.norm3 = norm_layer(self.filters[2])
        self.norm4 = norm_layer(self.filters[3])
        self.norm5 = norm_layer(self.filters[4])

        self.dnorm1 = norm_layer(self.filters[3])
        self.dnorm2 = norm_layer(self.filters[2])
        self.dnorm3 = norm_layer(self.filters[1])
        self.dnorm4 = norm_layer(self.filters[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=self.filters[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=self.filters[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block3 = nn.ModuleList([shiftedBlock(
            dim=self.filters[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block4 = nn.ModuleList([shiftedBlock(
            dim=self.filters[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block5 = nn.ModuleList([shiftedBlock(
            dim=self.filters[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=self.filters[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=self.filters[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock3 = nn.ModuleList([shiftedBlock(
            dim=self.filters[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock4 = nn.ModuleList([shiftedBlock(
            dim=self.filters[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.decoder1 = nn.Conv2d(self.filters[4], self.filters[3], 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(self.filters[3], self.filters[2], 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(self.filters[2], self.filters[1], 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(self.filters[1], self.filters[0], 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(self.filters[0], self.filters[0], 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(self.filters[3])
        self.dbn2 = nn.BatchNorm2d(self.filters[2])
        self.dbn3 = nn.BatchNorm2d(self.filters[1])
        self.dbn4 = nn.BatchNorm2d(self.filters[0])


        # gain low_feats

        self.conv_7 = nn.Conv2d(32, 4, kernel_size=7, stride=1, padding=3, bias=False)
        #self.conv_1 = nn.Conv2d(4, 128, kernel_size=1, stride=1, padding=0, bias=False)


        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)

        self.final = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        self.final1 = nn.Conv2d(32, num_classes, kernel_size=1)
        # MSA
        self.MSA1 = MSA(dim=self.filters[0], scales=2)
        self.MSA2 = MSA(dim=self.filters[1], scales=3)
        self.MSA3 = MSA(dim=self.filters[2], scales=4)
        self.MSA4 = MSA(dim=self.filters[3], scales=5)
        self.MSA5 = MSA(dim=self.filters[4], scales=6)
        # CSA
        self.CSA1 = CSA(dim=self.filters[0],mlp_ratio=4, drop_path=0)#,mlp_ratio=4, drop_path=0
        self.CSA2 = CSA(dim=self.filters[1],mlp_ratio=4, drop_path=0)
        self.CSA3 = CSA(dim=self.filters[2],mlp_ratio=4, drop_path=0)
        self.CSA4 = CSA(dim=self.filters[3],mlp_ratio=4, drop_path=0)
        self.CSA5 = CSA(dim=self.filters[4],mlp_ratio=4, drop_path=0)
        # CA
        self.att1 = DSE(self.filters[0])
        self.att2 = DSE(self.filters[1])
        self.att3 = DSE(self.filters[2])
        self.att4 = DSE(self.filters[3])
        self.att5 = DSE(self.filters[4])
        # FD
        self.sizes = [img_size // 2, img_size // 4, img_size // 8, img_size // 16, img_size // 32]
        self.FD4 = GlobalFilter(self.filters[3],out_channels=self.filters[3])

        # refine
        self.local_avg = nn.AvgPool2d(7, stride=1, padding=3)
        self.local_max = nn.MaxPool2d(7, stride=1, padding=3)
        self.local_convFM = nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1)
        self.local_ResMM = ResBlock(1, 1)
        self.local_pred2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1)
        self.proj = nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1)
        self.local_pred3 = nn.Conv2d(1, 1, kernel_size=1, padding=0, stride=1)
        self.local_pred4 = nn.Conv2d(1, 1, kernel_size=1, padding=0, stride=1)

        self.refine = nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1)


        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)  # 4  16384 16
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        # print("111111111111111111")
        # print(out.shape)
        low_feats = self.Gain_Low_feature(out)
        out = self.norm1(out)  #128  128

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()#torch.Size([8, 16, 256, 256])

        # CA
        out = self.att1(out)#torch.Size([8, 16, 256, 256])
        t1 = out

        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))

        out = self.MSA2(out)

        out = self.CSA2(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att2(out)
        t2 = out

        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        out = self.MSA3(out)
        out = self.CSA3(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att3(out)
        t3 = out

        ### Stage 4
        out = F.relu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        out = self.MSA4(out)
        # print(out.shape)
        out = self.CSA4(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block4):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att4(out)

        t4 = out
        t4 = self.FD4(t4)

        ### Bottleneck(5)
        out = F.relu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        out = self.MSA5(out)
        # print(out.shape)
        out = self.CSA5(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block5):
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att5(out)

        ### Stage 4
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)

        out = self.MSA4(out)
        out = self.CSA4(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = self.dnorm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att4(out)

        ### Stage 3
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)

        out = self.MSA3(out)
        out = self.CSA3(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att3(out)

        ## Stage 2
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)

        out = self.MSA2(out)
        out = self.CSA2(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock3):
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att2(out)

        ### Stage 1
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)

        out = self.MSA1(out)
        out = self.CSA1(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock4):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.att1(out)

        ### Stage 0
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        #print(out.shape)
        b, c, h, w = out.size()    #1 16 256 256
        # out = out.resize(b, int(c / 4), int(h * 2), int(w * 2))    # 1 128 512 512
        # print(out.shape)
        final = self.final1(out)      #1 1 512 512
        #print(final.shape)
        score = final
        final = self.Local_refinement(score, low_feats)
        #print(final.shape)  # 1 1 512 512
        return final

import time
if __name__ == '__main__':


    model = CTM-UNet(1).cuda()

    myinput = torch.zeros((1, 3, 512, 512)).cuda()

    flops,params = profile(model,inputs = (myinput,))
    start_time = time.time()
    end_time = time.time()
    sum_time = end_time - start_time
    print("infer_time:{:.3f}ms".format(sum_time * 1000))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
    print(flops, params)





