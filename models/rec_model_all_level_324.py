from cv2 import normalize
import cv2
import torch
from torch import nn
import random
from models.rec_models import sp_encoder, gl_encoder, generator_324, discriminator 
import loss.rec_loss.loss as loss
from loss.public_loss.lpips.lpips import LPIPS
from torchvision import utils, transforms


class RecModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lambda_R1 = 10.0
        self.lambda_L1 = 1.0
        self.lambda_Lpips = 1.0
        self.lambda_GAN = 1.0

        self.E_SP = sp_encoder.StyleGAN2ResnetSpEncoder(input_c=3)
        self.E_GL = gl_encoder.StyleGAN2ResnetGlEncoder(input_c=3)
        self.G = generator_324.StyleGAN2ResnetGenerator()
        self.D = discriminator.StyleGAN2Discriminator(size=args.output_size)

        self.register_buffer(
            "num_discriminator_iters", torch.zeros(1, dtype=torch.long)
        )
        self.l1_loss = torch.nn.L1Loss()
        self.lpips_loss = LPIPS(net_type='vgg', device=args.device).to(args.device).eval()


    def compute_image_discriminator_losses(self, real, rec):

        pred_real = self.D(real)
        pred_rec = self.D(rec)

        losses = {}
        losses["D_real"] = loss.gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.lambda_GAN

        losses["D_rec"] = loss.gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (0.5 * self.lambda_GAN)

        return losses


    def compute_discriminator_losses(self, real, E2, E3, E4, E5, G2, G4, G5):
        to_tensor_2 = transforms.ToTensor()
        self.num_discriminator_iters.add_(1)

        image_edge, ori_imag, mask, y_hat = real
        mask = 1 - mask
        image_edge = image_edge[:, 0:3, :, :]
        delta_edit = image_edge - y_hat
        min_val = delta_edit.min()
        max_val = delta_edit.max()
        delta_edit = (delta_edit - min_val) / (max_val - min_val)
        

        sp = self.E_SP(y_hat) 
        skip = E2(image_edge) 
        skip2 = E3(image_edge) 
        skip3 = E4(image_edge) 
        skip5 = E5(image_edge) 
        skip32 = torch.cat((skip3, skip5), dim=1) 
        gl, sp_style = self.E_GL(delta_edit) 
        rec = self.G(sp, gl, sp_style, skip, skip2, skip32, G2, G4, G5)
        losses = self.compute_image_discriminator_losses(ori_imag.detach(), rec.detach())

        
        #if randn == 1 or randn == 4:
         #   losses = self.compute_image_discriminator_losses(ori, rec)
        #else:
         #   losses = self.compute_image_discriminator_losses(ori_edit, rec)

        return losses


    def compute_R1_loss(self, real):
        image_edge, ori_imag, mask, y_hat = real
        real = ori_imag #real['ori']
        losses = {}
        real.requires_grad_()
        pred_real = self.D(real).sum()
        grad_real, = torch.autograd.grad(
            outputs=pred_real,
            inputs=[real],
            create_graph=True,
            retain_graph=True,
        )
        grad_real2 = grad_real.pow(2)
        dims = list(range(1, grad_real2.ndim))
        grad_penalty = grad_real2.sum(dims) * (self.lambda_R1 * 0.5)

        losses["D_R1"] = grad_penalty

        return losses


    def compute_generator_losses(self, real, E2, E3, E4, E5, G2, G4, G5):
        losses = {}

        image_edge, ori_imag, mask, y_hat = real
        mask = 1 - mask
        image_edge = image_edge[:, 0:3, :, :]
        delta_edit = image_edge - y_hat
        min_val = delta_edit.min()
        max_val = delta_edit.max()
        delta_edit = (delta_edit - min_val) / (max_val - min_val)

        sp = self.E_SP(y_hat)
        skip = E2(image_edge)
        skip2 = E3(image_edge)
        skip3 = E4(image_edge)
        skip5 = E5(image_edge)
        skip32=torch.cat((skip3, skip5), dim=1)
        gl, sp_style = self.E_GL(delta_edit)
        rec = self.G(sp, gl, sp_style, skip, skip2, skip32, G2, G4, G5)

        losses["G_L1"] = self.l1_loss(rec, ori_imag) * self.lambda_L1
        losses["G_Lpips"] = self.lpips_loss(rec, ori_imag) * self.lambda_Lpips

        #if randn == 1 or randn == 4:
           # losses["G_L1"] = self.l1_loss(rec, ori) * self.lambda_L1
           # losses["G_Lpips"] = self.lpips_loss(rec, ori) * self.lambda_Lpips
        #else:
           # losses["G_L1"] = self.l1_loss(rec, ori_edit) * self.lambda_L1
           # losses["G_Lpips"] = self.lpips_loss(rec, ori_edit) * self.lambda_Lpips
        
        losses["G_GAN_rec"] = loss.gan_loss(
            self.D(rec),
            should_be_classified_as_real=True
        ) * (self.lambda_GAN * 0.5)
        losses["G_GAN_rec"] = losses["G_GAN_rec"].mean()

        return losses, rec, delta_edit


    def encode_decode(self, delta_edit, e4e_edit):
        sp = self.E_SP(e4e_edit)
        gl, sp_style = self.E_GL(delta_edit)
        return self.G(sp, gl, sp_style)


    def get_parameters_for_mode(self, mode):
        if mode == "generator":
            return list(self.G.parameters()) + list(self.E_SP.parameters()) + list(self.E_GL.parameters())
        elif mode == "discriminator":
            Dparams = []
            Dparams += list(self.D.parameters())
            return Dparams
