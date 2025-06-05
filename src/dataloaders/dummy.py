from dataclasses import dataclass

import torch
from torchvision.transforms.v2 import Compose

from .base import BaseDataset
from .build import DATASET_REGISTRY
from .tensor_cache import TensorCache
from .types import DatasetOutput


@dataclass
class DummyDatasetConfig:
    path: str = "data/dummy.csv"


@DATASET_REGISTRY.register()
class DummyDataset(BaseDataset):
    def __init__(
        self,
        cfg: DummyDatasetConfig,
        transforms: Compose | None = None,
        cache: TensorCache | None = None,
    ) -> None:
        self.cfg = DummyDatasetConfig(**cfg)
        self.transforms = transforms
        self.cache = cache

    def __len__(self) -> int:
        return 32

    def __getitem__(self, idx: int) -> DatasetOutput:
        return DatasetOutput(data=torch.ones(8), label=idx % 4)
