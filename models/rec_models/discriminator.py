from .base_network import BaseNetwork
from models.rec_models.stylegan2_layers import Discriminator as OriginalStyleGAN2Discriminator


class StyleGAN2Discriminator(BaseNetwork):

    def __init__(self, size):
        super().__init__()
        use_antialias = True
        self.stylegan2_D = OriginalStyleGAN2Discriminator(
            size,
            2.0 * self.netD_scale_capacity,
            blur_kernel=[1, 3, 3, 1] if use_antialias else [1]
        )

    def forward(self, x):
        pred = self.stylegan2_D(x)
        return pred

    def get_features(self, x):
        return self.stylegan2_D.get_features(x)

    def get_pred_from_features(self, feat, label):
        assert label is None
        feat = feat.flatten(1)
        out = self.stylegan2_D.final_linear(feat)
        return out


