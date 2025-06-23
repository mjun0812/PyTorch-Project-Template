from typing import NotRequired, TypedDict

from torch import Tensor


class DatasetOutput(TypedDict, total=False):
    """Output structure for dataset samples.

    Attributes:
        data: The input data tensor (optional).
        label: The target label as integer (optional).
    """

    data: NotRequired[Tensor]
    label: NotRequired[int]
