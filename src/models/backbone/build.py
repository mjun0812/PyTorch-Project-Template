import timm
import torch
import torchvision
from omegaconf import OmegaConf
from torchvision.models._utils import IntermediateLayerGetter

from ...config import BackboneConfig
from ...utils import Registry
from ..layers import FrozenBatchNorm2d

BACKBONE_REGISTRY = Registry("BACKBONE")

TIMM_MODEL_LIST = timm.list_models()


def get_available_backbones() -> list[str]:
    """
    選択可能なすべてのバックボーンモデルのリストを取得します

    Returns:
        list[str]: 利用可能なバックボーンモデル名のリスト
    """
    custom_models = list(BACKBONE_REGISTRY._obj_map.keys())

    timm_models = TIMM_MODEL_LIST

    torchvision_models = torchvision.models.list_models(module=torchvision.models)
    torchvision_models = [f"torchvision_{model}" for model in torchvision_models]

    all_models = custom_models + timm_models + torchvision_models
    return all_models


def build_backbone(cfg: BackboneConfig) -> tuple[torch.nn.Module, list[int]]:
    model_name = cfg.name
    args = cfg.args
    if args is None:
        args = {}
    else:
        args = OmegaConf.to_object(args)

    if model_name in BACKBONE_REGISTRY._obj_map.keys():
        backbone = BACKBONE_REGISTRY.get(cfg.name)(
            pretrained=cfg.pretrained,
            weight_path=cfg.pretrained_weight,
            **args,
        )
        backbone_num_channels = backbone.feature_channels
    elif model_name in TIMM_MODEL_LIST:
        backbone = timm.create_model(
            cfg.name,
            features_only=True,
            pretrained=cfg.pretrained,
            **args,
        )
        backbone_num_channels = backbone.feature_info.channels()
    elif model_name.startswith("torchvision_"):
        model_name = model_name.replace("torchvision_", "")

        # 重みの設定をシンプルに
        weights = "DEFAULT" if cfg.pretrained and cfg.pretrained_weight is None else None

        # チャネル数をディクショナリで定義
        resnet_channels = {
            "resnet18": [32, 64, 128, 256, 512],
            "resnet34": [32, 64, 128, 256, 512],
        }.get(model_name, [128, 256, 512, 1024, 2048])

        # return_layersとbackbone_num_channelsを一度に構築
        use_backbone_features = args.pop("out_indices", [])
        return_layers = {f"layer{idx}": str(i) for i, idx in enumerate(use_backbone_features)}
        backbone_num_channels = [resnet_channels[idx] for idx in use_backbone_features]

        # バックボーンを取得して返す
        norm_layer = FrozenBatchNorm2d if cfg.freeze else None
        backbone = getattr(torchvision.models, model_name)(
            weights=weights, norm_layer=norm_layer, **args
        )
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    else:
        raise ValueError(f"Model {model_name} not found")

    return backbone, backbone_num_channels
