import torch.nn as nn

from .backbone import build_backbone
from .build import MODEL_REGISTRY  # noqa


@MODEL_REGISTRY.register()
class BaseModel(nn.Module):
    def __init__(self, cfg, phase="train"):
        super().__init__()
        self.cfg = cfg
        self.phase = phase
        self.backbone, _ = build_backbone(self.cfg)

    def forward(self, x):
        return self.backbone(x["image"])[0]
