from dataclasses import dataclass, field
from typing import Optional

from .base import BaseConfig


@dataclass
class LossConfig(BaseConfig):
    name: str = "BaseLoss"
    loss: str = "BaseLoss"
    args: Optional[dict] = None


@dataclass
class ModelConfig(BaseConfig):
    name: str = "BaseModel"
    model: str = "BaseModel"

    pre_trained_weight: Optional[str] = None
    trained_weight: Optional[str] = None

    use_sync_bn: bool = False
    find_unused_parameters: bool = False

    loss: LossConfig = field(default_factory=LossConfig)
