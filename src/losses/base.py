import torch.nn as nn

from ..config import ExperimentConfig
from ..types import LossOutput


class BaseLoss(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, targets: dict, preds: dict) -> LossOutput:
        raise NotImplementedError
