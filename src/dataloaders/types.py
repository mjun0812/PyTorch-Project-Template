from typing import NotRequired, TypedDict

from torch import Tensor


class DatasetOutput(TypedDict, total=False):
    data: NotRequired[Tensor]
    label: NotRequired[int]
