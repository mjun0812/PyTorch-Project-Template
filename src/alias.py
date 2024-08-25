from pathlib import Path
from typing import Literal, NotRequired, Required, TypedDict, Union

import torch
from torch import Tensor

PhaseStr = Literal["train", "val", "test"]
PathLike = Union[str, Path]


class LossOutput(TypedDict, total=False):
    total_loss: Required[Tensor]


class ModelOutput(TypedDict, total=False):
    losses: NotRequired[LossOutput]
    preds: NotRequired[dict[str, Tensor]]


class DatasetOutput(TypedDict, total=False):
    data: NotRequired[Tensor]
    label: NotRequired[int]


TORCH_DTYPE = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}
