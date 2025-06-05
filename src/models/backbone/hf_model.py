from typing import Any

from timm import create_model
from torch.nn import Module

from .build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
def resnext101d_32x4d(pretrained: bool = True, **kwargs: dict[str, Any]) -> Module:
    model = create_model(
        "hf_hub:mjun0812/resnext101d_32x4d", pretrained=pretrained, features_only=True, **kwargs
    )
    model.feature_channels = model.feature_info.channels()
    return model
