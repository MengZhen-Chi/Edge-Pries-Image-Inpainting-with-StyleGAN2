import math
import torch
from utils import util
import torch.nn.functional as F
from .base_network import BaseNetwork
from models.rec_models.stylegan2_layers import ConvLayer, ToRGB, EqualLinear, StyledConv, StyledConvSp


class ResolutionPreservingResnetBlock(torch.nn.Module):
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


class UpsamplingResnetBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim, stylespdim, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.transpose2d = torch.nn.ConvTranspose2d(stylespdim, stylespdim, 3, stride=2, padding=1, output_padding=1)
        self.convsp = StyledConvSp(inch, inch, 3, stylespdim, upsample=False)
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True, blur_kernel=blur_kernel)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=True, bias=True)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style, sp_style):
        # add spatial modulated conv
        x = self.convsp(x, sp_style)
        sp_style = self.transpose2d(sp_style)
        skip = F.interpolate(self.skip(x), scale_factor=2, mode='bilinear', align_corners=False)
        res = self.conv2(self.conv1(x, style), style)

        return (skip + res) / math.sqrt(2), sp_style


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


class StyleGAN2ResnetGenerator(BaseNetwork):
    """ The Generator (decoder) architecture described in Figure 18
        1. SpatialCodeModulation: First of all, modulate the structure code 
        with the global code.
        2. HeadResnetBlock: resnets at the resolution of the structure code,
        which also incorporates modulation from the global code.
        3. UpsamplingResnetBlock: resnets that upsamples by factor of 2 until
        the resolution of the output RGB image, along with the global code
        modulation.
        4. ToRGB: Final layer that transforms the output into 3 channels (RGB).
    """

    def __init__(self):
        super().__init__()
        # ref encoder
        # num_upsamplings = 4
        use_antialias = True
        blur_kernel = [1, 3, 3, 1] if use_antialias else [1]

        self.global_code_ch = self.global_code_ch

        self.add_module(
            "SpatialCodeModulation",
            GeneratorModulation(self.global_code_ch, self.spatial_code_ch))
        
        #self.modify_feature_alignment = FeatureAlignment(in_channels=640, out_channels=512)
        #self.modify_feature_extraction = FeatureExtraction(in_channels=512, out_channels=512)

        in_channel = self.spatial_code_ch
        for i in range(self.netG_num_base_resnet_layers):
            # gradually increase the number of channels
            out_channel = (i + 1) / self.netG_num_base_resnet_layers * self.nf(0)
            out_channel = max(self.spatial_code_ch, round(out_channel))
            layer_name = "HeadResnetBlock%d" % i
            new_layer = ResolutionPreservingResnetBlock(
                in_channel, out_channel, self.global_code_ch, self.spatial_code_ch)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        for j in range(self.netE_num_downsampling_sp):
            out_channel = self.nf(j + 1)
            layer_name = "UpsamplingResBlock%d" % (2 ** (4 + j))
            new_layer = UpsamplingResnetBlock(
                in_channel, out_channel, self.global_code_ch, self.spatial_code_ch,
                blur_kernel)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        last_layer = ToRGB(out_channel, self.global_code_ch,
                           blur_kernel=blur_kernel)
        self.add_module("ToRGB", last_layer)

    def nf(self, num_up):
        ch = 128 * (2 ** (self.netE_num_downsampling_sp - num_up))
        ch = int(min(512, ch) * self.netG_scale_capacity)
        return ch

    def forward(self, spatial_code, global_code, sp_style, skip, skip2, skip3, skip5, G2, G3, G4, G5):
        spatial_code = util.normalize(spatial_code) # [8, 16, 16, 16]
        global_code = util.normalize(global_code) # [8, 2048]
        skip = util.normalize(skip)
        skip2 = util.normalize(skip2)

        # spatial_code(8,8,16,16), global_code(8, 2048)
        x = self.SpatialCodeModulation(spatial_code, global_code) # [8, 16, 16, 16] [8, 512, 16, 16]
        for i in range(self.netG_num_base_resnet_layers):
            resblock = getattr(self, "HeadResnetBlock%d" % i)
            x = resblock(x, global_code, sp_style)
        x = torch.cat((x, skip5), dim=1)
        x = G5(x, global_code)

        for j in range(self.netE_num_downsampling_sp):
            key_name = 2 ** (4 + j)
            upsampling_layer = getattr(self, "UpsamplingResBlock%d" % key_name)
            x, sp_style = upsampling_layer(x, global_code, sp_style) # [8, 512, 32, 32] [8, 16, 32, 32] [8, 512, 64, 64] [8, 16, 64, 64] [8, 256, 128, 128] [8, 16, 128, 128] [8, 128, 256, 256] [8, 16, 256, 256]
            if j == 1:
                x = torch.cat((x, skip), dim=1)
                x = G2(x, global_code)
                #x = self.modify_feature_extraction(x)
            if j == 2:
                x = torch.cat((x, skip2), dim=1)
                x = G3(x, global_code)
            if j == 0:
                x = torch.cat((x, skip3), dim=1)
                x = G4(x, global_code)
            
        rgb = self.ToRGB(x, global_code, None) # [8, 3, 256, 256]

        return rgb
    """StyleGAN2ResnetSpEncoder(
  (FromRGB): ConvLayer(
    (Conv): EqualConv2d(3, 32, 1, stride=1, padding=0)
    (Act): FusedLeakyReLU()
  )
  (DownToSpatialCode): Sequential(
    (ResBlockDownBy1): ResBlock(
      (conv1): ConvLayer(
        (RefPad): ReflectionPad2d((1, 1, 1, 1))
        (Conv): EqualConv2d(32, 32, 3, stride=1, padding=0)
        (Act): FusedLeakyReLU()
      )
      (conv2): ConvLayer(
        (Blur): Blur(
          (reflection_pad): ReflectionPad2d((2, 1, 2, 1))
        )
        (Conv): EqualConv2d(32, 64, 3, stride=2, padding=0)
        (Act): FusedLeakyReLU()
      )
      (skip): ConvLayer(
        (Blur): Blur()
        (Conv): EqualConv2d(32, 64, 1, stride=2, padding=0)
      )
    )
    (ResBlockDownBy2): ResBlock(
      (conv1): ConvLayer(
        (RefPad): ReflectionPad2d((1, 1, 1, 1))
        (Conv): EqualConv2d(64, 64, 3, stride=1, padding=0)
        (Act): FusedLeakyReLU()
      )
      (conv2): ConvLayer(
        (Blur): Blur(
          (reflection_pad): ReflectionPad2d((2, 1, 2, 1))
        )
        (Conv): EqualConv2d(64, 128, 3, stride=2, padding=0)
        (Act): FusedLeakyReLU()
      )
      (skip): ConvLayer(
        (Blur): Blur()
        (Conv): EqualConv2d(64, 128, 1, stride=2, padding=0)
      )
    )
    (ResBlockDownBy4): ResBlock(
      (conv1): ConvLayer(
        (RefPad): ReflectionPad2d((1, 1, 1, 1))
        (Conv): EqualConv2d(128, 128, 3, stride=1, padding=0)
        (Act): FusedLeakyReLU()
      )
      (conv2): ConvLayer(
        (Blur): Blur(
          (reflection_pad): ReflectionPad2d((2, 1, 2, 1))
        )
        (Conv): EqualConv2d(128, 256, 3, stride=2, padding=0)
        (Act): FusedLeakyReLU()
      )
      (skip): ConvLayer(
        (Blur): Blur()
        (Conv): EqualConv2d(128, 256, 1, stride=2, padding=0)
      )
    )
    (ResBlockDownBy8): ResBlock(
      (conv1): ConvLayer(
        (RefPad): ReflectionPad2d((1, 1, 1, 1))
        (Conv): EqualConv2d(256, 256, 3, stride=1, padding=0)
        (Act): FusedLeakyReLU()
      )
      (conv2): ConvLayer(
        (Blur): Blur(
          (reflection_pad): ReflectionPad2d((2, 1, 2, 1))
        )
        (Conv): EqualConv2d(256, 512, 3, stride=2, padding=0)
        (Act): FusedLeakyReLU()
      )
      (skip): ConvLayer(
        (Blur): Blur()
        (Conv): EqualConv2d(256, 512, 1, stride=2, padding=0)
      )
    )
  )
)"""
