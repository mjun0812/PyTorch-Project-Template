import torch.nn as nn

from ..alias import ModelOutput, PhaseStr
from ..config import ExperimentConfig
from ..losses import build_loss


class BaseModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig, phase: PhaseStr = "train"):
        super().__init__()
        self.cfg = cfg
        self.phase = phase

        self.loss = None
        if self.phase in ["train", "val"]:
            self.loss = build_loss(self.cfg)

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
