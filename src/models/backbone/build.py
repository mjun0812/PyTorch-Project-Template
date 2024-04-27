import timm
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from ...utils import Registry
from ..layers import FrozenBatchNorm2d
from .internimage import create_model as create_internimage_model

BACKBONE_REGISTRY = Registry("BACKBONE")


# BACKBONE: resnet50
#   BACKBONE_ARGS:
#     out_indices: [1, 2, 3, 4]
#   IMAGENET_PRE_TRAINED: true
#   USE_BACKBONE_FEATURES: [1, 2, 3, 4]


def build_backbone(cfg):
    args = cfg.MODEL.get("BACKBONE_ARGS", {})
    model_name = cfg.MODEL.BACKBONE

    if model_name.startswith("torchvision_"):
        model_name = model_name.replace("torchvision_", "")

        weights = "DEFAULT" if cfg.MODEL.get("IMAGENET_PRE_TRAINED", True) else None

        if model_name in ("resnet18", "resnet34"):
            resnet_backbone_num_channels = [32, 64, 128, 256, 512]
        else:
            resnet_backbone_num_channels = [128, 256, 512, 1024, 2048]

        return_layers = {}
        backbone_num_channels = []
        for i, num_feat in enumerate(cfg.MODEL.USE_BACKBONE_FEATURES):
            return_layers[f"layer{num_feat}"] = str(i)
            backbone_num_channels.append(resnet_backbone_num_channels[num_feat])

        backbone = getattr(torchvision.models, model_name)(
            weights=weights, norm_layer=FrozenBatchNorm2d, **args
        )
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    elif model_name.startswith("internimage"):
        backbone = create_internimage_model(
            model_name,
            features_only=True,
            pretrained=cfg.MODEL.get("IMAGENET_PRE_TRAINED", True),
            **args,
        )
        backbone_num_channels = [info["num_chs"] for info in backbone.feature_info]
    elif model_name in BACKBONE_REGISTRY._obj_map.keys():
        backbone = BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE)(
            pretrained=cfg.MODEL.IMAGENET_PRE_TRAINED,
            weight_path=cfg.MODEL.IMAGENET_PRE_TRAINED_WEIGHT,
            **args,
        )
        backbone_num_channels = backbone.feature_channels
    else:
        backbone = timm.create_model(
            cfg.MODEL.BACKBONE,
            features_only=True,
            pretrained=cfg.MODEL.get("IMAGENET_PRE_TRAINED", True),
            **args,
        )
        backbone_num_channels = backbone.feature_info.channels()

    return backbone, backbone_num_channels
