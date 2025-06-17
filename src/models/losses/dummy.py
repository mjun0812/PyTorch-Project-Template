import torch.nn.functional as F

from ...dataloaders import DatasetOutput
from ..types import LossOutput
from .base import BaseLoss
from .build import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class DummyLoss(BaseLoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, targets: DatasetOutput, preds: dict) -> LossOutput:
        loss = F.nll_loss(preds["pred"], targets["label"])
        return LossOutput(total_loss=loss)
