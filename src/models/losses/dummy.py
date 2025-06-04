from dataclasses import dataclass

import torch.nn.functional as F

from ...config import LossConfig
from ...dataloaders.types import DatasetOutput
from ...types import LossOutput
from .base import BaseLoss
from .build import LOSS_REGISTRY


@dataclass
class DummyLossConfig(LossConfig):
    weight: float = 1.0


@LOSS_REGISTRY.register()
class DummyLoss(BaseLoss):
    def __init__(self, cfg: LossConfig):
        super().__init__(cfg)
        self.cfg = DummyLossConfig(**cfg.args)

    def forward(self, targets: DatasetOutput, preds: dict) -> LossOutput:
        loss = F.nll_loss(preds["pred"], targets["label"])
        return LossOutput(total_loss=loss)
