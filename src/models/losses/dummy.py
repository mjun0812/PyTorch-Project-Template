from dataclasses import dataclass

import torch.nn.functional as F

from ...dataloaders import DatasetOutput
from ..types import LossOutput
from .base import BaseLoss
from .build import LOSS_REGISTRY


@dataclass
class DummyLossConfig:
    weight: float = 1.0


@LOSS_REGISTRY.register()
class DummyLoss(BaseLoss):
    """Dummy loss function for testing and development.

    Implements a simple negative log-likelihood loss for classification tasks.
    This is primarily used for testing the loss computation pipeline.

    Attributes:
        cfg: Configuration containing loss weight and other parameters.
    """

    def __init__(self, cfg: dict | None) -> None:
        """Initialize the dummy loss function.

        Args:
            cfg: Configuration dictionary containing loss parameters.
        """
        super().__init__(cfg)
        self.cfg = DummyLossConfig(**(cfg or {}))

    def forward(self, targets: DatasetOutput, preds: dict) -> LossOutput:
        """Compute the dummy loss.

        Args:
            targets: Target data containing ground truth labels.
            preds: Model predictions containing logits or probabilities.

        Returns:
            LossOutput containing the computed negative log-likelihood loss.
        """
        loss = F.nll_loss(preds["pred"], targets["label"])
        return LossOutput(total_loss=loss)
