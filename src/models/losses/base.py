from torch import nn

from ...config import LossConfig
from ...types import LossOutput


class BaseLoss(nn.Module):
    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()

    def forward(self, targets: dict, preds: dict) -> LossOutput:
        raise NotImplementedError
