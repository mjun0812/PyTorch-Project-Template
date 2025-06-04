from pathlib import Path
from typing import Literal

import torch

PhaseStr = Literal["train", "val", "test"]
PathLike = str | Path


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
