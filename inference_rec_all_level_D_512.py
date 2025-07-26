import argparse
from sympy import li
import torch
from qqdm import qqdm
from models.rec_model_all_level_324 import RecModel
import torchvision
import wandb
from models.psp import pSp
from dataset.images_dataset import ImagesDataset
from torch.utils.data import DataLoader
from add_encoder.E1 import *
from models.rec_models import sp_encoder2, sp_encoder3, sp_encoder4, sp_encoder5_324
from models.rec_models.stylegan2_layers import StyledConv


class Trainer(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.test_dataset = ImagesDataset(args)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.args.test_batch,
                                          shuffle=False,
                                          num_workers=int(self.args.test_workers),
                                          drop_last=True)
        
        self.model = RecModel(args).to(self.args.device)
        ckpt = None
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location='cpu')
            self.model.E_GL.load_state_dict(ckpt['E_GL'], strict=True)
            self.model.E_SP.load_state_dict(ckpt['E_SP'], strict=True)
            self.model.G.load_state_dict(ckpt['G'], strict=True)
            self.model.D.load_state_dict(ckpt['D'], strict=True)


        self.E_SP2 = sp_encoder2.StyleGAN2ResnetSpEncoder(input_c=3)
        self.E_SP2 = self.E_SP2.to(self.args.device)
        self.E_SP2.load_state_dict(ckpt['ESP2'], strict=True)

        self.E_SP3 = sp_encoder3.StyleGAN2ResnetSpEncoder(input_c=3)
        self.E_SP3 = self.E_SP3.to(self.args.device)
        self.E_SP3.load_state_dict(ckpt['ESP3'], strict=True)

        self.E_SP4 = sp_encoder4.StyleGAN2ResnetSpEncoder(input_c=3)
        self.E_SP4 = self.E_SP4.to(self.args.device)
        self.E_SP4.load_state_dict(ckpt['ESP4'], strict=True)

        self.E_SP5 = sp_encoder5_324.StyleGAN2ResnetSpEncoder(input_c=3)
        self.E_SP5 = self.E_SP5.to(self.args.device)
        self.E_SP5.load_state_dict(ckpt['ESP5'], strict=True)

        self.G2 = StyledConv(in_channel=320, out_channel=256, kernel_size=3, style_dim=2048, upsample=False, use_noise=True)
        self.G2 = self.G2.to(self.args.device)
        self.G2.load_state_dict(ckpt['G2'], strict=True)


        self.G4 = StyledConv(in_channel=640, out_channel=512, kernel_size=3, style_dim=2048, upsample=False, use_noise=True)
        self.G4 = self.G4.to(self.args.device)
        self.G4.load_state_dict(ckpt['G4'], strict=True)

        self.G5 = StyledConv(in_channel=1280, out_channel=512, kernel_size=3, style_dim=2048, upsample=False, use_noise=True)
        self.G5 = self.G5.to(self.args.device)
        self.G5.load_state_dict(ckpt['G5'], strict=True)

        self.net = pSp(args).to(args.device)
        self.net.eval()
        self.model.eval()
        self.E_SP2.eval()
        self.G2.eval()
        self.E_SP3.eval()
        self.E_SP4.eval()
        self.G4.eval()
        self.E_SP5.eval()
        self.G5.eval()


    
    

    def val(self):
        lpi = 0.0
        for batch_idx, batch in enumerate(self.test_dataloader):
            batch[0] = batch[0].to(self.args.device)
            batch[1] = batch[1].to(self.args.device)
            batch[2] = batch[2].to(self.args.device)
            image_edge, ori_imag, mask, len = batch
            y_hat, _ = self.net.forward(image_edge, return_latents=True)
            batch.append(y_hat)
            g_losses, rec, delta_edit = self.train_one_step(batch, self.E_SP2, self.E_SP3, 
                                                                self.E_SP4, self.E_SP5, self.G2, 
                                                                self.G4, self.G5)
            lpi += g_losses['G_Lpips'].item()
            torchvision.utils.save_image(rec, f"{args.log_image}/{str(batch_idx).zfill(6)}.png")
        print('Training finished.')
        print(lpi / len)

    def validate(self, id):
        self.E_SP2.eval()
        self.E_SP3.eval()
        self.G2.eval()
        self.E_SP4.eval()
        self.G4.eval()
        self.E_SP5.eval()
        self.G5.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            batch[0] = batch[0].to(self.args.device)
            batch[1] = batch[1].to(self.args.device)
            batch[2] = batch[2].to(self.args.device)
            image_edge, ori_imag, mask = batch
            y_hat, _ = self.net.forward(image_edge, return_latents=True)
            batch.append(y_hat)
            cur_loss_dict = {}
            real_data = {'ori':ori_imag,
                        'ori_edge':image_edge,
                        'y_hat':y_hat}
            with torch.no_grad():
                loss, rec, delta_edit = self.model.compute_generator_losses(batch, self.E_SP2, self.E_SP3, 
                                                                            self.E_SP4, self.E_SP5, self.G2, 
                                                                            self.G4, self.G5)
                cur_loss_dict = {'val':loss}
            agg_loss_dict.append(cur_loss_dict)
            self.saveImages(batch_idx, real_data, rec, delta_edit, title='test')

            # For first step just do sanity test on small amount of data
            if id == 0 and batch_idx >= 4:
                self.E_SP2.train()
                self.G2.train()
                self.E_SP3.train()
                #self.G3.train()
                self.E_SP4.train()
                self.G4.train()
                self.E_SP5.train()
                self.G5.train()
                return None  # Do not log, inaccurate in first batch

        #loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.print_metrics(cur_loss_dict, prefix='test', global_step=batch_idx)

        self.model.train()
        return agg_loss_dict
    
    def print_metrics(self, metrics_dict, prefix, global_step=0):
        print('Metrics for {}, step {}'.format(prefix, global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)


    def set_requires_grad(self, params, requires_grad):
        for p in params:
            p.requires_grad_(requires_grad)


    def train_one_step(self, images_minibatch, E2, E3, E4, E5, G2, G4, G5):
        #d_losses = self.train_discriminator_one_step(images_minibatch, E2, E3, E4, E5, G2, G4, G5)
        g_losses, rec, delta_edit = self.train_generator_one_step(images_minibatch, E2, E3, E4, E5, G2, G4, G5)
        
        return g_losses, rec, delta_edit


    def train_generator_one_step(self, images, E2, E3, E4, E5, G2, G4, G5):
        #self.set_requires_grad(self.Dparams, False)
        #self.set_requires_grad(self.Gparams, True)
        #self.optimizer_G.zero_grad()
        g_losses, rec, delta_edit = self.model.compute_generator_losses(images, E2, E3, E4, E5, G2, G4, G5)
        #g_loss = sum([v.mean() for v in g_losses.values()])
        #g_loss.backward()
        #self.optimizer_G.step()

        return g_losses, rec, delta_edit


    def train_discriminator_one_step(self, images, E2, E3, E4, E5, G2, G4, G5):
        if self.model.lambda_GAN == 0.0:
            return {}
        self.set_requires_grad(self.Dparams, True)
        self.set_requires_grad(self.Gparams, False)
        self.discriminator_iter_counter += 1
        self.optimizer_D.zero_grad()
        d_losses = self.model.compute_discriminator_losses(images, E2, E3, E4, E5, G2, G4, G5)
        d_loss = sum([v.mean() for v in d_losses.values()])
        d_loss.backward()
        self.optimizer_D.step()

        needs_R1 = self.model.lambda_R1 > 0.0
        needs_R1_at_current_iter = needs_R1 and \
            self.discriminator_iter_counter % self.R1_once_every == 0
        if needs_R1_at_current_iter:
            self.optimizer_D.zero_grad()
            r1_losses = self.model.compute_R1_loss(images)
            d_losses.update(r1_losses)
            r1_loss = sum([v.mean() for v in r1_losses.values()])
            r1_loss = r1_loss * self.R1_once_every
            r1_loss.backward()
            self.optimizer_D.step()

        d_losses["D_total"] = sum([v.mean() for v in d_losses.values()])

        return d_losses
        

    def saveCkpt(self, iBatch):
        if iBatch % 7500 == 0:
            state = {'epoch'     : iBatch,
                     'ESP2'      : self.E_SP2.state_dict(), 
                     'ESP3'      : self.E_SP3.state_dict(),
                     'ESP4'      : self.E_SP4.state_dict(),
                     'ESP5'      : self.E_SP5.state_dict(),   
                     'G2'        : self.G2.state_dict(),
                     'G4'        : self.G4.state_dict(),
                     'G5'        : self.G5.state_dict(),
                     'D'         : self.model.D.state_dict(),
                     'G'         :self.model.G.state_dict(),
                     'E_GL'      :self.model.E_GL.state_dict(),
                     'E_SP'      :self.model.E_SP.state_dict(),
                     'optimizer_G' : self.optimizer_G.state_dict(),
                     'optimizer_D' : self.optimizer_D.state_dict(),
                     }
            save_file = f"{self.args.save_ckpt}/{iBatch}.pth"
            torch.save(state, save_file)
            print(f"... saving checkpoint after epoch {iBatch} ...")
            print(f"... saving path : {save_file} ...")


    def saveImages(self, iBatch, real, rec, delta_edit, title):
        vb = self.args.batch // 2
        ori, e4e_ori = real['ori'], real['ori_edge'][:, 0:3, :, :]
        ori_edit = real['y_hat']
        concat_images = torch.cat(
            [ori[:vb], e4e_ori[:vb], ori_edit[:vb], \
                delta_edit[:vb], rec[:vb]], dim = 0)
        torchvision.utils.save_image(
            concat_images, 
            f"{args.log_image}/{title}/{str(iBatch).zfill(6)}.png",
            nrow = args.batch // 2)


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description = '...stylegan2 editing...')
    parser.add_argument('--batch', default = 8, type = int)
    parser.add_argument('--test_batch', default = 1, type = int)
    parser.add_argument('--workers', default=8, type=int, help='Number of train dataloader workers')
    parser.add_argument('--test_workers', default=1, type=int,
                                 help='Number of test/inference dataloader workers')
    parser.add_argument('--device', default = 'cuda:0', type = str) 
    parser.add_argument('--epoch', default = 1, type = int)
    parser.add_argument('--size', default = 1024, type = int)
    parser.add_argument('--input_size', default = 256, type = int)
    parser.add_argument('--output_size', default = 256, type = int)
    parser.add_argument('--latent_dim', default = 512, type = int)
    parser.add_argument('--nb_layer', default = 18, type = int)
    parser.add_argument('--report_frequency', default = 100, type = int)
    parser.add_argument('--seed', default = 6230, type = int)
    parser.add_argument('--truncation', default = 1, type = float)
    parser.add_argument('--start_from_latent_avg', default = False, action = 'store_true')
    parser.add_argument('--save_ckpt', default = '', type = str, help='path to save the training weights') 
    parser.add_argument('--log_image', default = '', type = str, help='path to store images') 
    parser.add_argument('--config', default = 'configs/adult.yaml', type = str)
    parser.add_argument('--edit_config', default = 'edit_configs/adult/basic_rec.yaml', type = str)
    parser.add_argument('--e4e_model', default='', type = str)
    parser.add_argument('--load_stylegan2', default = '', type = str, help='path to Edge-e4e generator images')
    parser.add_argument('--finetune', default='', type=str)
    parser.add_argument('--checkpoint_path', 
                        default='', 
                        type=str, help='Path to Edge-e4e encoder checkpoint') #************
    parser.add_argument('--encoder_type', default='Encoder4Editing', type=str, help='Which encoder to use')
    parser.add_argument('--stylegan_size', default=1024, type=int,
                        help='size of pretrained StyleGAN Generator')
    parser.add_argument('--test_source_root', default='', type=str, help='path to the test images')
    parser.add_argument('--test_mask_root', default='', type=str, help='path to the test mask images')
    parser.add_argument('--test_edge_root', default='', type=str, help='path to the test edge images')
    parser.add_argument('--source_root', default='', type=str, help='path to the training images')
    parser.add_argument('--mask_root', default='', type=str, help='path to the training mask images')
    parser.add_argument('--edge_root', default='', type=str, help='path to the training edge images')
    parser.add_argument('--board_interval', default=100, type=int,
                                 help='Interval for logging metrics to tensorboard')
    parser.add_argument('--val_interval', default=8750, type=int, help='Validation interval')
    parser.add_argument('--test', default=True, type=str, help='building test datasets')
    parser.add_argument('--save_interval', default=100, type=int, help='save image interval')
    parser.add_argument('--ckpt', default='', type=str, help='pretrained model path') #**********
    
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.val() 