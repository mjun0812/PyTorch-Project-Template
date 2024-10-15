from dataclasses import dataclass
from typing import Optional

from .config import BaseConfig


@dataclass
class BackboneConfig(BaseConfig):
    name: str
    imagenet_pretrained: bool = True
    imagenet_pretrained_weight: Optional[str] = None
    use_backbone_features: Optional[list] = None
    freeze_backbone: bool = False
    args: Optional[dict] = None
