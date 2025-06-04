from typing import NotRequired, Required, TypedDict

from torch import Tensor


class LossOutput(TypedDict, total=False):
    total_loss: Required[Tensor]


class ModelOutput(TypedDict, total=False):
    losses: NotRequired[LossOutput]
    preds: NotRequired[dict[str, Tensor]]
