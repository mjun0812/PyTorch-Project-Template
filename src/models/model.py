import torch.nn as nn

from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class Model(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, x):
        return x
