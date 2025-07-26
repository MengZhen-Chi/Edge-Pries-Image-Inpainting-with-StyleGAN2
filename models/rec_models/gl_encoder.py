import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import util
from .base_network import BaseNetwork
from models.rec_models.stylegan2_layers import ResBlock, ConvLayer, EqualLinear


class StyleGAN2ResnetGlEncoder(BaseNetwork):

    def __init__(self, input_c):
        super().__init__()
        use_antialias = True

        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1, 2, 1] if use_antialias else [1]

        self.add_module("FromRGB", ConvLayer(input_c, self.nc(0), 1))

        self.DownToSpatialCode = nn.Sequential()
        for i in range(self.netE_num_downsampling_sp):
            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel, reflection_pad=True)
            )

        nchannels = self.nc(self.netE_num_downsampling_sp)
        self.add_module(
            "ToSpatialCode",
            nn.Sequential(
                ConvLayer(nchannels, nchannels, 1, activate=True, bias=True),
                ConvLayer(nchannels, self.spatial_code_ch, kernel_size=1,
                          activate=False, bias=True)
            )
        )
        
        self.DownToGlobalCode = nn.Sequential()
        for i in range(self.netE_num_downsampling_gl):
            idx_from_beginning = self.netE_num_downsampling_sp + i
            self.DownToGlobalCode.add_module(
                "ConvLayerDownBy%d" % (2 ** idx_from_beginning),
                ConvLayer(self.nc(idx_from_beginning),
                          self.nc(idx_from_beginning + 1), kernel_size=3,
                          blur_kernel=[1], downsample=True, pad=0)
            )
            
        nchannels = self.nc(self.netE_num_downsampling_sp +
                            self.netE_num_downsampling_gl)
        self.add_module(
            "ToGlobalCode",
            nn.Sequential(
                EqualLinear(nchannels, self.global_code_ch)
            )
        )

    def nc(self, idx):
        nc = self.netE_nc_steepness ** (5 + idx)
        nc = nc * self.netE_scale_capacity
        nc = min(self.global_code_ch, int(round(nc)))
        return round(nc)


    def forward(self, x):
        x = self.FromRGB(x)
        midpoint = self.DownToSpatialCode(x)
        sp = self.ToSpatialCode(midpoint)

        x = self.DownToGlobalCode(midpoint)
        x = x.mean(dim=(2, 3))
        gl = self.ToGlobalCode(x)
        gl = util.normalize(gl)

        return gl, sp

