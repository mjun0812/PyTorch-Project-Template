import random
from typing import NotRequired, TypedDict

import torch
from torch import Tensor


class DatasetOutput(TypedDict, total=False):
    """Output structure for dataset samples.

    Attributes:
        data: The input data tensor (optional).
        label: The target label as integer (optional).
    """

    # GPUに転送するキーのリスト
    __gpu_keys__: tuple[str] = ("data", "label")

    data: NotRequired[Tensor]
    label: NotRequired[int]

    @classmethod
    def dummy(cls, batch: int | None = None) -> "DatasetOutput":
        """Create dummy DatasetOutput for testing.

        Args:
            batch: Batch size. If None, returns single sample with label as int.
                   If specified, returns batched data with label as Tensor.

        Returns:
            DatasetOutput with dummy values.
        """
        if batch is None:
            return cls(data=torch.randn(8), label=random.randint(0, 1000))
        else:
            return cls(
                data=torch.randn(batch, 8), label=[random.randint(0, 1000) for _ in range(batch)]
            )
