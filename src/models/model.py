import torch.nn as nn

from .build import MODEL_REGISTRY  # noqa


# @MODEL_REGISTRY.register()
class BaseModel(nn.Module):
    def __init__(self, cfg, phase):
        super().__init__()
        self.cfg = cfg
        self.phase = phase

    def forward(self, x, data=None):
        return x
