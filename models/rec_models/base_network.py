import torch


class BaseNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.netE_scale_capacity = 1.0
        self.netE_num_downsampling_sp = 4
        self.netE_num_downsampling_gl = 2
        self.netE_nc_steepness = 2.0

        self.netG_scale_capacity = 1.0
        self.netG_num_base_resnet_layers = 2
        self.netG_use_noise = True
        self.netG_resnet_ch = 256

        self.netD_scale_capacity = 1.0

        self.global_code_ch = 2048
        # self.global_spatial_code_ch = 8
        self.spatial_code_ch = 16
        # self.num_classes = 100


    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def collect_parameters(self, name):
        params = []
        for m in self.modules():
            if type(m).__name__ == name:
                params += list(m.parameters())
        return params

    def fix_and_gather_noise_parameters(self):
        params = []
        device = next(self.parameters()).device
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                assert m.image_size is not None, "One forward call should be made to determine size of noise parameters"
                m.fixed_noise = torch.nn.Parameter(torch.randn(m.image_size[0], 1, m.image_size[2], m.image_size[3], device=device))
                params.append(m.fixed_noise)
        return params

    def remove_noise_parameters(self, name):
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                m.fixed_noise = None

    def forward(self, x):
        return x
