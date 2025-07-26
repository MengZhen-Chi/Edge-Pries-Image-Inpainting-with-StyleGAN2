import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def gan_loss(pred, should_be_classified_as_real):
    bs = pred.size(0)
    if should_be_classified_as_real:
        return F.softplus(-pred).view(bs, -1).mean(dim=1)
    else:
        return F.softplus(pred).view(bs, -1).mean(dim=1)


def feature_matching_loss(xs, ys, equal_weights=False, num_layers=6):
    loss = 0.0
    for i, (x, y) in enumerate(zip(xs[:num_layers], ys[:num_layers])):
        if equal_weights:
            weight = 1.0 / min(num_layers, len(xs))
        else:
            weight = 1 / (2 ** (min(num_layers, len(xs)) - i))
        loss = loss + (x - y).abs().flatten(1).mean(1) * weight
    return loss
