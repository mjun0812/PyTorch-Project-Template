import torch.nn as nn

from ..config import ModelConfig
from ..types import ModelOutput, PhaseStr
from .losses import build_loss


class BaseModel(nn.Module):
    def __init__(self, cfg: ModelConfig, phase: PhaseStr = "train"):
        super().__init__()
        self.cfg = cfg
        self.phase = phase

        self.loss = None
        if self.phase in ["train", "val"]:
            self.loss = build_loss(self.cfg.loss)

    def train_forward(self, data: dict) -> ModelOutput:
        raise NotImplementedError

    def val_forward(self, data: dict) -> ModelOutput:
        raise NotImplementedError

    def test_forward(self, data: dict) -> ModelOutput:
        raise NotImplementedError

    def forward(self, data: dict) -> ModelOutput:
        if self.phase == "train":
            return self.train_forward(data)
        elif self.phase == "val":
            return self.val_forward(data)
        else:
            return self.test_forward(data)
