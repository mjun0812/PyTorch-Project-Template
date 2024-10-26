from dataclasses import dataclass, field
from typing import Optional

from .base import BaseConfig


@dataclass
class TransformConfig(BaseConfig):
    name: str = "BaseTransform"
    args: Optional[dict] = None


@dataclass
class DatasetConfig(BaseConfig):
    name: str = "BaseDataset"
    dataset: str = "BaseDataset"

    train_transforms: list[TransformConfig] = field(default_factory=list)
    train_batch_transforms: Optional[list[TransformConfig]] = None
    val_transforms: list[TransformConfig] = field(default_factory=list)
    test_transforms: list[TransformConfig] = field(default_factory=list)
