import torch
from torch import nn


class WNormLoss(nn.Module):

	def __init__(self, netConfig):
		super().__init__()
		self.real_init(
			start_from_latent_avg = netConfig["start_from_latent_avg"]
		)


	def real_init(self, start_from_latent_avg=True):
		self.start_from_latent_avg = start_from_latent_avg


	def forward(self, data):
		return self.real_forward(
			latent     = data['mWp'],
			latent_avg = data['latentAvg']
		)


	def real_forward(self, latent, latent_avg=None):
		if self.start_from_latent_avg:
			latent = latent - latent_avg
		ret =  torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]
		# import ipdb; ipdb.set_trace()
		return ret
