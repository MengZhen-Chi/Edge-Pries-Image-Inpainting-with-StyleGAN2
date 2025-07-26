import torch
import torch.nn as nn
import torch.nn.functional as F


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



class FeatureAlignment(nn.Module):
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

    def forward(self, x):

        #x = torch.cat((encoder_feats,generator_feats), dim=1)
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

class FeatureExtraction(nn.Module):
    def __init__(self,in_channels, out_channels ):
        super().__init__() 
        t_channel = 512
        self.first_layer = nn.Conv2d(in_channels, t_channel, kernel_size=1, padding=0, bias=True)
        self.convs = nn.Sequential(*[ResLayers(t_channel,t_channel,1), ResLayers(t_channel,out_channels,1), ResLayers(out_channels,out_channels,1) ])

    def forward(self, aligned_feats):
        y = self.first_layer(aligned_feats)
        y = self.convs(y)
        return y
    


class GateNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        t_channel = 256
        self.down1 = nn.Conv2d(in_channels, t_channel, kernel_size=3, padding=1, bias=True)
        self.down2 = nn.Conv2d(in_channels, t_channel, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.convs = nn.Sequential(*[ResLayers(in_channels,in_channels,1), ResLayers(in_channels,out_channels,1), ResLayers(out_channels,out_channels,1) ])
        self.convs2 = nn.Sequential(*[ResLayers(in_channels,in_channels,1), ResLayers(in_channels,out_channels,1), ResLayers(out_channels,1,1) ])


    def forward(self, generator_feats, y):
        generator_feats = self.down1(generator_feats)
        y = self.down2(y)
        x = torch.cat((generator_feats, y), dim=1)
        deltaF = self.convs(x)
        gate = self.convs2(x)
        gate = self.sigmoid(gate)
        return deltaF, gate