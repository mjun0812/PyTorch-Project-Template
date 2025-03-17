from timm import create_model

from .build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
def resnext101d_32x4d(pretrained: bool = True, **kwargs):
    model = create_model("hf_hub:mjun0812/resnext101d_32x4d", pretrained=pretrained, **kwargs)
    return model
