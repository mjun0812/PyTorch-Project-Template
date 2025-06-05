from torch.utils.data import Dataset

from .tensor_cache import TensorCache


class BaseDataset(Dataset):
    def __init__(self, cfg: dict, transforms=None, cache: TensorCache | None = None):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError
