import torch.nn.functional as F

from ...config import ExperimentConfig
from ...types import LossOutput
from .base import BaseLoss
from .build import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class DummyLoss(BaseLoss):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)

    def forward(self, targets: dict, preds: dict) -> LossOutput:
        loss = F.nll_loss(preds["pred"], targets["label"])
        return LossOutput(total_loss=loss)
