import math
from utils import util
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stylegan2.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from models.rec_models.stylegan2_layers import ConvLayer, ToRGB, EqualLinear, StyledConv, StyledConvSp


class DownResnetblock(torch.nn.Module):
    def __init__(self, inch, outch, styledim, stylespdim):
        super().__init__()
        self.convsp = StyledConvSp(inch, inch, 3, stylespdim, upsample=False)
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=False)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=False, bias=False)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style, sp_style):
        # add spatial modulated conv
        x = self.convsp(x, sp_style)
        skip = self.skip(x)
        res = self.conv2(self.conv1(x, style), style)

        return (skip + res) / math.sqrt(2)

class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ResLayers(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        if (in_channels == out_channels) and stride==1:
            self.shortcut_layer = nn.Identity()
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride, bias=False))
        
        self.res_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=True), nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=True) )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class ResidualFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        t_channel = 512
        self.first_layer = nn.Conv2d(in_channels, t_channel, kernel_size=1, padding=0, bias=True)

        self.conv1 =  nn.Sequential(*[ResLayers(t_channel,t_channel,1)])
        self.conv2 =  nn.Sequential(*[ResLayers(t_channel,t_channel,2), ResLayers(t_channel,t_channel,1)])
        self.conv3 =  nn.Sequential(*[ResLayers(t_channel,t_channel,2), ResLayers(t_channel,t_channel,1)])

        self.dconv1 = nn.Sequential(*[ResLayers(t_channel,t_channel,1), ResLayers(t_channel,t_channel,1)])
        self.dconv2 = nn.Sequential(*[ResLayers(t_channel,t_channel,1), ResLayers(t_channel,t_channel,1)])
        self.dconv3 = nn.Sequential(*[ResLayers(t_channel,t_channel,1), ResLayers(t_channel,t_channel,1)])

        self.out_layer = nn.Conv2d(t_channel, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, encoder_feats, generator_feats):

        x = torch.cat((encoder_feats,generator_feats), dim=1)
        x = self.first_layer(x)

        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        shape = f3.shape[-1]
        df1 = F.interpolate(f3, size=(shape*2,shape*2) , mode='bilinear', align_corners=True)
        df2 = self.dconv1(df1 + f2)
        df2 =  F.interpolate(df2, size=(shape*4,shape*4) , mode='bilinear', align_corners=True)
        df3 = self.dconv2(df2 + f1)

        aligned_feats = self.out_layer(df3)

        return aligned_feats
    

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        t_channel = 512
        self.first_layer = nn.Conv2d(in_channels, t_channel, kernel_size=1, padding=0, bias=True)
        self.convs = nn.Sequential(*[ResLayers(t_channel,t_channel,1), ResLayers(t_channel,out_channels,1), ResLayers(out_channels,out_channels,1) ])

    def forward(self, aligned_feats):
        y = self.first_layer(aligned_feats)
        y = self.convs(y)
        return y

class GeneratorModulation(torch.nn.Module):
    def __init__(self, styledim, outch):
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        if style.ndimension() <= 2:
            return x * (1 * self.scale(style)[:, :, None, None]) + self.bias(style)[:, :, None, None]
        else:
            style = F.interpolate(style, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            return x * (1 * self.scale(style)) + self.bias(style)

class upper(nn.Module):
    def __init__(self, in_channels=2048, mid_channels=16, out_channels=256):
        super().__init__()
        self.two_equal = GeneratorModulation(in_channels, mid_channels)
        self.downRes = DownResnetblock(mid_channels, out_channels, 2048, 16)
        self.up_conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.training_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(32, 15), stride=(32, 15), padding=0, groups=512)
        self.linear_layer = nn.Linear(1024, 18)
        #self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, lp, sp, gl):
        lp = util.normalize(lp)
        sp = util.normalize(sp)

        x = self.two_equal(lp, sp) # [8, 16, 32, 32]
        x = self.downRes(x, sp, gl) # [8, 256, 32, 32]
        x = self.up_conv(x) # [8, 512, 32, 32]
        x = x.reshape(8*512, 32*32)
        x = self.linear_layer(x)
        x = x.view(8, 18, 512)

        return x
