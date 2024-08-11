from typing import Optional

import torch
from torch.utils.data import Dataset

from ..alias import PhaseStr
from ..config import ExperimentConfig
from .build import DATASET_REGISTRY
from .tensor_cache import TensorCache


@DATASET_REGISTRY.register()
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


@DATASET_REGISTRY.register()
class DummyDataset(BaseDataset):
    def __init__(
        self,
        cfg: ExperimentConfig,
        transforms=None,
        phase: PhaseStr = "train",
        cache: Optional[TensorCache] = None,
    ):
        super().__init__(cfg, transforms, phase, cache)

    def __len__(self):
        return 10

    def __getitem__(self, idx) -> dict:
        return dict(image=torch.ones(256), label=idx % 10)
