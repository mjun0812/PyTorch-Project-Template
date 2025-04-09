from typing import Optional

from torch.utils.data import Dataset

from .tensor_cache import TensorCache


class BaseDataset(Dataset):
    def __init__(self, cfg: dict, transforms=None, cache: Optional[TensorCache] = None):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx) -> dict:
        raise NotImplementedError
