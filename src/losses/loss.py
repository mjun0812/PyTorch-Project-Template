import torch.nn.functional as F

from .build import LOSS_REGISTRY


@LOSS_REGISTRY.register()
def loss(x):
    return F.mse_loss(x)
