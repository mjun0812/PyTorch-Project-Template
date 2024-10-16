import timm
import torch
import torchvision
from omegaconf import OmegaConf
from torchvision.models._utils import IntermediateLayerGetter

from ...config import BackboneConfig
from ...utils import Registry
from ..layers import FrozenBatchNorm2d
from .internimage import create_model as create_internimage_model

BACKBONE_REGISTRY = Registry("BACKBONE")


def build_backbone(cfg: BackboneConfig) -> tuple[torch.nn.Module, list[int]]:
    cfg_backbone = BackboneConfig(**cfg)

    model_name = cfg_backbone.name
    args = cfg_backbone.args
    if args is None:
        args = {}
    else:
        args = OmegaConf.to_object(args)

    if model_name.startswith("torchvision_"):
        model_name = model_name.replace("torchvision_", "")

        weights = None
        if cfg_backbone.pretrained and cfg_backbone.pretrained_weight is None:
            weights = "DEFAULT"

        if model_name in ("resnet18", "resnet34"):
            resnet_backbone_num_channels = [32, 64, 128, 256, 512]
        else:
            resnet_backbone_num_channels = [128, 256, 512, 1024, 2048]

        return_layers = {}
        backbone_num_channels = []
        use_backbone_features = args.pop("out_indices", [])
        for i, num_feat in enumerate(use_backbone_features):
            return_layers[f"layer{num_feat}"] = str(i)
            backbone_num_channels.append(resnet_backbone_num_channels[num_feat])

        backbone = getattr(torchvision.models, model_name)(
            weights=weights,
            norm_layer=FrozenBatchNorm2d if cfg_backbone.freeze else None,
            **args,
        )
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    elif model_name.startswith("internimage"):
        backbone = create_internimage_model(
            model_name,
            features_only=True,
            pretrained=cfg_backbone.pretrained,
            **args,
        )
        backbone_num_channels = [info["num_chs"] for info in backbone.feature_info]
    elif model_name in BACKBONE_REGISTRY._obj_map.keys():
        backbone = BACKBONE_REGISTRY.get(cfg_backbone.name)(
            pretrained=cfg_backbone.pretrained,
            weight_path=cfg_backbone.pretrained_weight,
            **args,
        )
        backbone_num_channels = backbone.feature_channels
    else:
        backbone = timm.create_model(
            cfg_backbone.name,
            features_only=True,
            pretrained=cfg_backbone.pretrained,
            **args,
        )
        backbone_num_channels = backbone.feature_info.channels()

    # freeze
    if cfg_backbone.freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, backbone_num_channels
