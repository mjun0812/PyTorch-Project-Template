from dataclasses import dataclass
from typing import Optional


@dataclass
class BackboneConfig:
    name: str
    pretrained: bool = True
    pretrained_weight: Optional[str] = None
    freeze: bool = False
    args: Optional[dict] = None
