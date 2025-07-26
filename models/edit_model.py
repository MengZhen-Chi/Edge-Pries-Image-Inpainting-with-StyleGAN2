import torch
import os
import yaml
from torch import nn
from models.networks.stylegan2.model import Generator
from models.networks.stylegan2.model import EqualLinear, PixelNorm


def convertAttribute(attrConfig, oldAttr, device='cuda'):

    eps = 1e-6
    nbBatch = oldAttr.shape[0]
    if attrConfig['typeEncoder'] == 'discrete':
        if oldAttr.dim() == 1:
            oldAttr = oldAttr.unsqueeze(1)
        index = oldAttr.long() # size(1, 1)
        if abs(oldAttr[0] - index[0]) < eps:
        # if True:
            # tensor([[1., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
            tensor  = torch.zeros(nbBatch, attrConfig['indim']).to(device).scatter_(dim = 1, index = index, value = 1) 
            # tensor *= 2
            return tensor
        else:
            delta   = oldAttr - index
            tensor1 = torch.zeros(nbBatch, attrConfig['indim']).to(device).scatter_(dim = 1, index = index, value = 1)
            tensor2 = torch.zeros(nbBatch, attrConfig['indim']).to(device).scatter_(dim = 1, index = index + 1, value = 1)
            return tensor1 * (1 - delta) + tensor2 * delta
    elif attrConfig['typeEncoder'] == 'continuous':
        converted = (oldAttr - attrConfig['convertShift']) / float(attrConfig['convertScale'])
        converted = converted.unsqueeze(dim = 1).float().to(device)
        return converted
    elif attrConfig['typeEncoder'] == 'embedding':
        return oldAttr
    else:
        raise Exception(f"ERROR: Unknown Encoder tyle: {attrConfig['typeEncoder']}")


class EqualMLP(nn.Module):


    def __init__(self, output_dim=512, input_nc=1, lr_mlp=0.01, nb_layer = 4):
        super(EqualMLP, self).__init__()
        layers = []
        layers.append(EqualLinear( input_nc, output_dim, lr_mul=lr_mlp, activation='fused_lrelu' ))
        for i in range(nb_layer - 1):
            layers.append(
                EqualLinear(
                    output_dim, output_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)


    def forward(self, x):
        n = x.size(0)
        style = self.style(x.view(n, -1))
        return style


class AttributeModulation(nn.Module):
    def __init__(self, output_channel, input_channel = 2):
        super().__init__()
        self.output_channel = output_channel
        self.mlp            = EqualMLP(input_nc = input_channel)
        self.activation     = nn.Tanh()
        self.fc_multiply    = nn.Linear(512, output_channel)
        self.fc_bias        = nn.Linear(512, output_channel)
    

    def forward(self, at, x):
        at      = self.mlp(at)
        at      = self.activation(at)
        at_mul  = self.fc_multiply(at)
        at_bias = self.fc_bias(at)

        out     = torch.sigmoid(at_mul) * x + at_bias
        return out



class AttentionGenerator(nn.Module):

    def __init__(self, latent_dim = 512, output_dim =512):
        super().__init__()
        self.Wk = nn.Linear(latent_dim, output_dim)
        self.Wq = nn.Linear(latent_dim, output_dim)
        self.Wv = nn.Linear(latent_dim, output_dim)


    def forward(self, proxyCode):
        Q = self.Wq(proxyCode)
        K = self.Wk(proxyCode)
        V = self.Wv(proxyCode)
        return (Q, K, V)



class ContinuousAttributeEncoder(nn.Module):

    def __init__(self, latent_dim = 512, output_dim = 512, L = 10):
        super().__init__()
        self.L          = L
        self.latent_dim = latent_dim
        self.attrFc     = nn.Linear(1, latent_dim)
        # self.attrFc     = nn.Linear(2 * L, latent_dim)
        self.maskFc     = nn.Linear(latent_dim, latent_dim)
        self.finaFc     = EqualMLP(output_dim = output_dim, input_nc = latent_dim, nb_layer = 3)
        self.attention  = AttentionGenerator(latent_dim, output_dim)


    def positionalEncode(self, attr, L = 10):
        pi     = 3.1415926
        output = []
        for i in range(L):
            output.append(torch.sin(2**i * pi * attr))
            output.append(torch.cos(2**i * pi * attr))
        return torch.cat(output, dim = -1)


    def forward(self, latent, attr):
        attrCode = self.attrFc(attr)
        maskCode = self.maskFc(latent)
        concCode = attrCode * maskCode
        finaCode = self.finaFc(concCode)
        return self.attention(finaCode)
        # return finaCode


class DiscreteAttributeEncoder(nn.Module):

    def __init__(self, latent_dim = 512, output_dim = 512, attr_channel = 1): 
        super().__init__()
        self.latent_dim = latent_dim
        self.attrFc     = nn.Linear(attr_channel, latent_dim)
        self.maskFc     = nn.Linear(latent_dim, latent_dim)
        self.finaFc     = EqualMLP(output_dim = output_dim, input_nc = latent_dim, nb_layer = 3)
        self.attention  = AttentionGenerator(latent_dim, output_dim)

    
    def forward(self, latent, attr):
        #! input attr should be one-hot
        attrCode = self.attrFc(attr)
        maskCode = self.maskFc(latent)
        concCode = attrCode * maskCode
        finaCode = self.finaFc(concCode)
        return self.attention(finaCode)
        # return finaCode


class EmbeddingEncoder(nn.Module):

    def __init__(self, latent_dim = 512, output_dim = 512, attr_channel = 364): 
        super().__init__()
        self.latent_dim = latent_dim
        self.attrFc     = nn.Linear(attr_channel, latent_dim)
        self.maskFc     = nn.Linear(latent_dim, latent_dim)
        self.finaFc     = EqualMLP(output_dim = output_dim, input_nc = latent_dim, nb_layer = 3)
        self.attention  = AttentionGenerator(latent_dim, output_dim)

    def forward(self, latent, attr):
        #! input attr should be one-hot
        attrCode = self.attrFc(attr)
        maskCode = self.maskFc(latent)
        concCode = attrCode * maskCode
        finaCode = self.finaFc(concCode)
        return self.attention(finaCode)


def attentiveFusion(proxyAttentions):

    Qs = torch.stack([ qkv[0] for qkv in proxyAttentions ]).permute(1, 0, 2) # b x a x 512
    Ks = torch.stack([ qkv[1] for qkv in proxyAttentions ]).permute(1, 2, 0) # b x 512 x a
    Vs = torch.stack([ qkv[2] for qkv in proxyAttentions ]).permute(1, 0, 2) # b x a x 512
    As = Qs.bmm(Ks).softmax(dim = 1)  # b x a x a
    Rs = As.bmm(Vs).sum(dim = 1) # b x a x 512
    return Rs


def naiveFusion(proxyAttentions):
    # return sum( qkv[-1] for qkv in proxyAttentions )
    return sum( qkv for qkv in proxyAttentions )


class JointEdit(nn.Module):

    def parseEnables(self):
        netConfig = self.netConfig
        attrEncoders  = {}
        setLayers     = set() 
        for attr in netConfig.keys():
            title = netConfig[attr]['title']
            setLayers |= set(netConfig[attr]['layers'])
            for idLayer in netConfig[attr]['layers']:
                keyEncoder = f'{title}@{idLayer}'
                if netConfig[attr]['typeEncoder'] == 'discrete':
                    attrEncoders[keyEncoder] = DiscreteAttributeEncoder(self.latentDim, self.latentDim, netConfig[attr]['indim'])
                elif netConfig[attr]['typeEncoder'] == 'continuous':
                    attrEncoders[keyEncoder] = ContinuousAttributeEncoder(self.latentDim, self.latentDim)
                elif netConfig[attr]['typeEncoder'] == 'embedding':
                    attrEncoders[keyEncoder] = EmbeddingEncoder( self.latentDim,  self.latentDim, attr_channel=364)
                else:
                    raise Exception(f"ERROR: Unknown Encoder type: {netConfig[attr]['typeEncoder']}")
        self.attrEncoders = nn.ModuleDict(attrEncoders)

        attrModulations = {}
        for idLayer in setLayers:
            attrModulations[f'{idLayer}'] = AttributeModulation(self.latentDim, self.latentDim)
        self.attrModulations = nn.ModuleDict(attrModulations)


    def __init__(self, nbLayer, latentDim, configFile):
        super().__init__()
        self.nbLayer   = nbLayer
        self.latentDim = latentDim

        postfix = os.path.splitext(configFile)[-1]
        if postfix == '.yaml':
            with open(configFile) as fs:
                self.netConfig = yaml.load(fs, Loader=yaml.FullLoader)['ATTR']
        elif postfix == '.pth':
            ckpt = torch.load(configFile)
            self.netConfig = ckpt['netConfig']
        else:
            raise Exception("ERROR: configFile suffix must be .yaml or .pth!")
        self.parseEnables()
        
        if postfix == '.pth':
            self.load_state_dict( ckpt['net'] )


    def forward(self, latentWp, attrTarg, attrFlag):
        latentDict = {}
        for keyEncoder, itemEncoder in self.attrEncoders.items():
            title, idLayer = keyEncoder.split('@')
            if title not in attrFlag or attrFlag[title] == 0: continue
            if idLayer not in latentDict:
                latentDict[f'{idLayer}'] = list()
            encode = itemEncoder(latentWp[:, int(idLayer), :], attrTarg[title]) #! note!
            latentDict[f'{idLayer}'].append( encode )

        self.proxyCodes = latentDict

        newLatent = []
        for idLayer in range(self.nbLayer):
            if f'{idLayer}' in latentDict:
                layerCode   = attentiveFusion(latentDict[f'{idLayer}'])
                # layerCode   = naiveFusion(latentDict[f'{idLayer}'])
                layerLatent = self.attrModulations[f'{idLayer}'](layerCode, latentWp[:, idLayer, :])
                newLatent.append(layerLatent)
            else:
                newLatent.append(latentWp[:, idLayer, :])
        newLatent = torch.cat(newLatent, dim = -1).view(latentWp.shape[0], -1, self.latentDim)
        return newLatent


class JointModel(nn.Module):
    
    
    def __init__(self, args):
        super().__init__()
        self.args      = args
        self.edit      = JointEdit(args.nb_layer, args.latent_dim, args.config)
        self.generator = Generator(args.size, args.latent_dim, 8, channel_multiplier = 2)
        self.avgpool   = nn.AdaptiveAvgPool2d((args.output_size, args.output_size))
        # self.avgpool   = nn.AdaptiveAvgPool2d((512, 512))
        self.load_pretrained()


    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.args.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
            else:
                self.latent_avg = None


    def load_pretrained(self):
        saved_state_dict = torch.load(self.args.load_stylegan2, map_location='cpu')
        self.generator.load_state_dict(saved_state_dict["g_ema"])
        for parameter in self.generator.parameters():
            parameter.requires_grad = False
        self.generator.eval()
        self.__load_latent_avg(saved_state_dict, self.generator.n_latent)
    

    def edit_from_wp(self, latent_wp, attr_targ, attr_flag, truncation = 1, truncation_latent = None, noise = [None] * 18):
        u_wp = latent_wp
        if self.args.start_from_latent_avg:
            u_wp = u_wp + self.latent_avg
        m_wp = self.edit(u_wp, attr_targ, attr_flag)
        self.u_wp = u_wp
        self.m_wp = m_wp
        m_image = self.generator.gen_from_latent(m_wp, noise)
        u_image = self.generator.gen_from_latent(u_wp, noise)
        if self.args.output_size != 1024:
            m_image, u_image = self.avgpool(m_image), self.avgpool(u_image)
        return m_image, u_image, m_wp, u_wp


    def generate_from_latent(self, latent, truncation=1, truncation_latent=None): 
        if latent.dim() == 2:  # latent z
            u_wp, noise = self.generator.get_latent_noise([latent], truncation = truncation, truncation_latent = truncation_latent)
        elif latent.dim() == 3:   # latent wp
            u_wp, noise = latent, [None] * 18
            
        if self.args.start_from_latent_avg:
            u_wp = u_wp + self.latent_avg
        u_image = self.generator.gen_from_latent(u_wp, noise)

        return u_image


    def forward(self, latent, attr_targ, attr_flag, truncation=1, truncation_latent=None, s_boundary=None):
        if latent.dim() == 2:
            u_wp, noise = self.generator.get_latent_noise([latent], truncation = truncation, truncation_latent = truncation_latent)
        elif latent.dim() == 3:
            u_wp, noise = latent, [None] * 18
        
        if s_boundary is not None:
            if self.args.start_from_latent_avg:
                u_wp = u_wp + self.latent_avg
            u_image = self.generator.gen_from_latent(u_wp, noise)
            m_image = self.generator.gen_from_latentw_s(u_wp, noise, s_boundary=s_boundary)
            if self.args.output_size != 1024:
                m_image, u_image = self.avgpool(m_image), self.avgpool(u_image)
            return m_image, u_image, u_wp, u_wp

        return self.edit_from_wp(u_wp, attr_targ, attr_flag, truncation, truncation_latent, noise)

