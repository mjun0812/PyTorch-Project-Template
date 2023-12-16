import torch.nn as nn

from .build import LOSS_REGISTRY  # noqa


# @LOSS_REGISTRY.register()
class BaseLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        return {"total_loss": x.mean()}
