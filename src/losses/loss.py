import torch.nn as nn
import torch.nn.functional as F

from .build import LOSS_REGISTRY


# @LOSS_REGISTRY.register()
class BaseLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        return {"total_loss": F.mse_loss(x)}
