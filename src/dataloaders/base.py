from typing import Any

from torch.utils.data import Dataset

from .tensor_cache import TensorCache
from .types import DatasetOutput


class BaseDataset(Dataset):
    def __init__(self, cfg: dict, transforms: Any = None, cache: TensorCache | None = None) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> DatasetOutput:
        raise NotImplementedError
