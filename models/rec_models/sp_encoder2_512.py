import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append('/home/liqing/Li/Li_qing/ReGANIE')
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
        self.netE_num_downsampling_sp = 3
        for i in range(self.netE_num_downsampling_sp):
            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel, reflection_pad=True)
            )



    def nc(self, idx):
        # Adjust channel progression to reach 128 at the final layer
        # Start with 32, then 64, then 128
        channel_sequence = [32, 64, 128, 128]  # Modified channel progression
        if idx < len(channel_sequence):
            return channel_sequence[idx]
        return self.global_code_ch  # default to max channels if beyond sequence

    def forward(self, x):
        x = self.FromRGB(x) # [8, 32, 256, 256]
        midpoint = self.DownToSpatialCode(x) # [8, 128, 64, 64]
        return midpoint