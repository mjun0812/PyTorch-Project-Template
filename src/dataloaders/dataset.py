from typing import Optional

import torch

from ..config import ExperimentConfig
from ..types import DatasetOutput, PhaseStr
from .base import BaseDataset
from .build import DATASET_REGISTRY
from .tensor_cache import TensorCache


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
        return 32

    def __getitem__(self, idx) -> dict:
        return DatasetOutput(data=torch.ones(1, 28, 28), label=idx % 10)
