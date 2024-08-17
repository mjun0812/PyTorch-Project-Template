from typing import Optional

from torch.utils.data import Dataset

from ..alias import PhaseStr
from ..config import ExperimentConfig
from .tensor_cache import TensorCache


class BaseDataset(Dataset):
    def __init__(
        self,
        cfg: ExperimentConfig,
        transforms=None,
        phase: PhaseStr = "train",
        cache: Optional[TensorCache] = None,
    ):
        self.cfg = cfg
        self.transforms = transforms
        self.phase = phase
        self.cache = cache

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx) -> dict:
        raise NotImplementedError
