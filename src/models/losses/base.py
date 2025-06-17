from torch import nn

from ..types import LossOutput


class BaseLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, targets: dict, preds: dict) -> LossOutput:
        raise NotImplementedError
