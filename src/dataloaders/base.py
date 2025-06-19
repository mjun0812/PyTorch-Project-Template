from abc import abstractmethod

from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose

from .tensor_cache import TensorCache
from .types import DatasetOutput


class BaseDataset(Dataset):
    def __init__(
        self,
        cfg: dict | None,
        transforms: Compose | None = None,
        cache: TensorCache | None = None,
    ) -> None:
        self.cfg = cfg
        self.transforms = transforms
        self.cache = cache

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> DatasetOutput:
        raise NotImplementedError
