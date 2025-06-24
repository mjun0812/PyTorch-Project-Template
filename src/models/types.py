from typing import NotRequired, Required, TypedDict

import torch
from torch import Tensor


class LossOutput(TypedDict, total=False):
    """Output structure for loss functions.

    Attributes:
        total_loss: The computed total loss tensor.
    """

    total_loss: Required[Tensor]

    @classmethod
    def dummy(cls) -> "LossOutput":
        """Create dummy LossOutput for testing."""
        return cls(total_loss=torch.rand(1))


class PredOutput(TypedDict, total=False):
    """Output structure for model predictions.

    Attributes:
        preds: Dictionary containing model predictions.
    """

    preds: NotRequired[Tensor]

    @classmethod
    def dummy(cls, batch: int = 1) -> "PredOutput":
        """Create dummy PredOutput for testing."""
        return cls(preds=torch.rand(batch, 10))


class ModelOutput(TypedDict, total=False):
    """Output structure for model forward passes.

    Attributes:
        losses: Dictionary containing computed losses (optional).
        preds: Dictionary containing model predictions (optional).
    """

    losses: NotRequired[LossOutput]
    preds: NotRequired[PredOutput]

    @classmethod
    def dummy(cls, batch: int = 1) -> "ModelOutput":
        """Create dummy ModelOutput for testing."""
        return cls(losses=LossOutput.dummy(), preds=PredOutput.dummy(batch))
