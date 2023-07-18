import torch.nn as nn

from ..losses import build_loss
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class BaseModel(nn.Module):
    def __init__(self, cfg, phase):
        self.cfg = cfg
        self.phase = phase

        self.criterion = build_loss(cfg)

    def forward(self, x, data=None):
        return x
