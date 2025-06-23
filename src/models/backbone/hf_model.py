from typing import Any

from timm import create_model
from torch.nn import Module

from .build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
def resnext101d_32x4d(pretrained: bool = True, **kwargs: dict[str, Any]) -> Module:
    """Create ResNeXt-101d 32x4d model from Hugging Face Hub.

    Args:
        pretrained: Whether to load pretrained weights.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        ResNeXt-101d model with feature extraction capabilities.
    """
    model = create_model(
        "hf_hub:mjun0812/resnext101d_32x4d", pretrained=pretrained, features_only=True, **kwargs
    )
    model.feature_channels = model.feature_info.channels()
    return model
