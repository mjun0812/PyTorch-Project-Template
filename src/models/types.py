from typing import NotRequired, Required, TypedDict

from torch import Tensor


class LossOutput(TypedDict, total=False):
    """Output structure for loss functions.

    Attributes:
        total_loss: The computed total loss tensor.
    """

    total_loss: Required[Tensor]


class ModelOutput(TypedDict, total=False):
    """Output structure for model forward passes.

    Attributes:
        losses: Dictionary containing computed losses (optional).
        preds: Dictionary containing model predictions (optional).
    """

    losses: NotRequired[LossOutput]
    preds: NotRequired[dict[str, Tensor]]
