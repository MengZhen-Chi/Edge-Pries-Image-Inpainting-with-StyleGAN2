import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import util
from .base_network import BaseNetwork
from models.rec_models.stylegan2_layers import ResBlock, ConvLayer


class StyleGAN2ResnetSpEncoder(BaseNetwork):

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

        # Spatial Code refers to the Structure Code, and
        # Global Code refers to the Texture Code of the paper.
        nchannels = self.nc(self.netE_num_downsampling_sp)
        self.add_module(
            "ToSpatialCode",
            nn.Sequential(
                ConvLayer(nchannels, nchannels, 1, activate=True, bias=True),
                ConvLayer(nchannels, self.spatial_code_ch, kernel_size=1,
                          activate=False, bias=True)
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
        sp = util.normalize(sp)

        return sp
