from dataclasses import dataclass
from typing import Optional

import torch
from torchvision.transforms.v2 import Compose

from ..types import DatasetOutput
from .base import BaseDataset
from .build import DATASET_REGISTRY
from .tensor_cache import TensorCache


@dataclass
class DummyDatasetConfig:
    path: str = "data/dummy.csv"


@DATASET_REGISTRY.register()
class DummyDataset(BaseDataset):
    def __init__(
        self,
        cfg: DummyDatasetConfig,
        transforms: Optional[Compose] = None,
        cache: Optional[TensorCache] = None,
    ):
        self.cfg = DummyDatasetConfig(**cfg)
        self.transforms = transforms
        self.cache = cache

    def __len__(self):
        return 32

    def __getitem__(self, idx) -> DatasetOutput:
        return DatasetOutput(data=torch.ones(8), label=idx % 4)
