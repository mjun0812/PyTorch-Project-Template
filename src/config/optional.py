from dataclasses import dataclass
from typing import Optional

from .config import BaseConfig


@dataclass
class BackboneConfig(BaseConfig):
    name: str
    pretrained: bool = True
    pretrained_weight: Optional[str] = None
    freeze: bool = False
    args: Optional[dict] = None
