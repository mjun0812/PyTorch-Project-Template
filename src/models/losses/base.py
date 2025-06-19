from torch import nn

from ..types import LossOutput


class BaseLoss(nn.Module):
    def __init__(self, cfg: dict | None) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, targets: dict, preds: dict) -> LossOutput:
        raise NotImplementedError
