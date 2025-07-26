import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append('/home/zhr/test/Rec')
from utils import util
from models.rec_models.base_network import BaseNetwork
from models.rec_models.stylegan2_layers import ResBlock, ConvLayer


class StyleGAN2ResnetSpEncoder(BaseNetwork):

    def __init__(self, input_c):
        super().__init__()
        use_antialias = True

        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1, 2, 1] if use_antialias else [1]

        self.add_module("FromRGB", ConvLayer(input_c, self.nc(0), 1))

        self.DownToSpatialCode = nn.Sequential()
        self.netE_num_downsampling_sp = 4
        for i in range(self.netE_num_downsampling_sp):
            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel, reflection_pad=True)
            )
        self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** 4),
                ResBlock(self.nc(4), self.nc(4), blur_kernel, reflection_pad=True, downsample=False)
            )




    def nc(self, idx):
        nc = self.netE_nc_steepness ** (5 + idx)
        nc = nc * self.netE_scale_capacity
        nc = min(self.global_code_ch, int(round(nc)))
        return round(nc)


    def forward(self, x):
        x = self.FromRGB(x) 
        midpoint = self.DownToSpatialCode(x) # 
        return midpoint