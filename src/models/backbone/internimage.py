import torch

from .build import BACKBONE_REGISTRY
from .internimage_impl import InternImage

DEFAULT_CONFIG = dict(
    core_op="DCNv3",
    num_classes=1000,
    channels=64,
    depths=[4, 4, 18, 4],
    groups=[4, 8, 16, 32],
    layer_scale=None,
    offset_scale=1.0,
    post_norm=False,
    mlp_ratio=4.0,
    res_post_norm=False,  # for InternImage-H/G
    dw_kernel_size=None,  # for InternImage-H/G
    use_clip_projector=False,  # for InternImage-H/G
    level2_post_norm=False,  # for InternImage-H/G
    level2_post_norm_block_ids=None,  # for InternImage-H/G
    center_feature_scale=False,  # for InternImage-H/G
    remove_center=False,
)


def get_config_internimage_t_1k_224(features_only, **kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=64,
            depths=[4, 4, 18, 4],
            groups=[4, 8, 16, 32],
            offset_scale=1.0,
            mlp_ratio=4.0,
        )
    )
    if features_only:
        config["layer_scale"] = 1.0
        config["drop_path_rate"] = 0.2
    config.update(kwargs)
    return config


def get_config_internimage_s_1k_224(features_only, **kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=80,
            depths=[4, 4, 21, 4],
            groups=[5, 10, 20, 40],
            layer_scale=1e-5,
            offset_scale=1.0,
            mlp_ratio=4.0,
            post_norm=True,
        )
    )
    if features_only:
        config["layer_scale"] = 1.0
        config["drop_path_rate"] = 0.3
    config.update(kwargs)
    return config


def get_config_internimage_b_1k_224(features_only, **kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=112,
            depths=[4, 4, 21, 4],
            groups=[7, 14, 28, 56],
            layer_scale=1e-5,
            offset_scale=1.0,
            post_norm=True,
            mlp_ratio=4.0,
        )
    )
    if features_only:
        config["layer_scale"] = 1.0
        config["drop_path_rate"] = 0.4
    config.update(kwargs)
    return config


def get_config_internimage_l_22kto1k_384(features_only, **kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=160,
            depths=[5, 5, 22, 5],
            groups=[10, 20, 40, 80],
            layer_scale=1e-5,
            offset_scale=2.0,
            post_norm=True,
            mlp_ratio=4.0,
        )
    )
    if features_only:
        config["layer_scale"] = 1.0
        config["drop_path_rate"] = 0.4
    config.update(kwargs)
    return config


def get_config_internimage_xl_22kto1k_384(features_only, **kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=192,
            depths=[5, 5, 24, 5],
            groups=[12, 24, 48, 96],
            layer_scale=1e-5,
            offset_scale=2.0,
            mlp_ratio=4.0,
            post_norm=True,
        )
    )
    if features_only:
        config["layer_scale"] = 1.0
        config["drop_path_rate"] = 0.4
    config.update(kwargs)
    return config


def get_config_internimage_h_22kto1k_384(features_only, **kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=320,
            depths=[6, 6, 32, 6],
            groups=[10, 20, 40, 80],
            layer_scale=None,
            offset_scale=1.0,
            post_norm=False,
            mlp_ratio=4.0,
            res_post_norm=True,  # for InternImage-H/G
            dw_kernel_size=5,  # for InternImage-H/G
            use_clip_projector=True,  # for InternImage-H/G
            level2_post_norm=True,  # for InternImage-H/G
            level2_post_norm_block_ids=[5, 11, 17, 23, 29],  # for InternImage-H/G
            center_feature_scale=True,  # for InternImage-H/G
        )
    )
    if features_only:
        config["drop_path_rate"] = 0.5
    config.update(kwargs)
    return config


def get_config_internimage_h_22kto1k_640(features_only, **kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=320,
            depths=[6, 6, 32, 6],
            groups=[10, 20, 40, 80],
            layer_scale=None,
            offset_scale=1.0,
            post_norm=False,
            mlp_ratio=4.0,
            res_post_norm=True,  # for InternImage-H/G
            dw_kernel_size=5,  # for InternImage-H/G
            use_clip_projector=True,  # for InternImage-H/G
            level2_post_norm=True,  # for InternImage-H/G
            level2_post_norm_block_ids=[5, 11, 17, 23, 29],  # for InternImage-H/G
            center_feature_scale=True,  # for InternImage-H/G
        )
    )
    if features_only:
        config["drop_path_rate"] = 0.5
    config.update(kwargs)
    return config


def get_config_internimage_g_22kto1k_512(features_only, **kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=512,
            depths=[2, 2, 48, 4],
            groups=[16, 32, 64, 128],
            layer_scale=None,
            offset_scale=1.0,
            post_norm=True,
            mlp_ratio=4.0,
            dw_kernel_size=5,  # for InternImage-H/G
            use_clip_projector=True,  # for InternImage-H/G
            level2_post_norm=True,  # for InternImage-H/G
            level2_post_norm_block_ids=[
                5,
                11,
                17,
                23,
                29,
                35,
                41,
                47,
            ],  # for InternImage-H/G
            center_feature_scale=True,  # for InternImage-H/G
        )
    )
    if features_only:
        config["drop_path_rate"] = 0.5
    config.update(kwargs)
    return config


MODELS = {
    "internimage_t_1k_224": {
        "config": get_config_internimage_t_1k_224,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth",
    },
    "internimage_s_1k_224": {
        "config": get_config_internimage_s_1k_224,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_s_1k_224.pth",
    },
    "internimage_b_1k_224": {
        "config": get_config_internimage_b_1k_224,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth",
    },
    "internimage_l_22kto1k_384": {
        "config": get_config_internimage_l_22kto1k_384,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22kto1k_384.pth",
    },
    "internimage_xl_22kto1k_384": {
        "config": get_config_internimage_xl_22kto1k_384,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22kto1k_384.pth",
    },
    "internimage_h_22kto1k_384": {
        "config": get_config_internimage_h_22kto1k_384,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_jointto22k_384.pth",
    },
    "internimage_h_22kto1k_640": {
        "config": get_config_internimage_h_22kto1k_640,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_22kto1k_640.pth",
    },
    "internimage_g_22kto1k_512": {
        "config": get_config_internimage_g_22kto1k_512,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_g_22kto1k_512.pth",
    },
}


def create_model(
    model_name: str,
    pretrained: bool = True,
    weight_path: str | None = None,
    features_only: bool = False,
    out_indices: list[int] | None = None,
    **kwargs,
):
    out_indices = out_indices or [0, 1, 2, 3]

    model_info = MODELS[model_name]
    config = model_info["config"](features_only, **kwargs)
    model = InternImage(**config, features_only=features_only, out_indices=out_indices)

    if pretrained:
        if weight_path is None:
            checkpoint = torch.hub.load_state_dict_from_url(
                url=model_info["pretrained"], map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(weight_path)
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def internimage_t_1k_224(
    pretrained, weight_path, features_only=True, out_indices=(0, 1, 2, 3), **kwargs
):
    model = create_model(
        "internimage_t_1k_224",
        pretrained=pretrained,
        weight_path=weight_path,
        features_only=features_only,
        out_indices=out_indices,
        **kwargs,
    )
    return model


@BACKBONE_REGISTRY.register()
def internimage_s_1k_224(
    pretrained, weight_path, features_only=True, out_indices=(0, 1, 2, 3), **kwargs
):
    model = create_model(
        "internimage_s_1k_224",
        pretrained=pretrained,
        weight_path=weight_path,
        features_only=features_only,
        out_indices=out_indices,
        **kwargs,
    )
    return model


@BACKBONE_REGISTRY.register()
def internimage_b_1k_224(
    pretrained, weight_path, features_only=True, out_indices=(0, 1, 2, 3), **kwargs
):
    model = create_model(
        "internimage_b_1k_224",
        pretrained=pretrained,
        weight_path=weight_path,
        features_only=features_only,
        out_indices=out_indices,
        **kwargs,
    )
    return model


@BACKBONE_REGISTRY.register()
def internimage_l_22kto1k_384(
    pretrained, weight_path, features_only=True, out_indices=(0, 1, 2, 3), **kwargs
):
    model = create_model(
        "internimage_l_22kto1k_384",
        pretrained=pretrained,
        weight_path=weight_path,
        features_only=features_only,
        out_indices=out_indices,
        **kwargs,
    )
    return model


@BACKBONE_REGISTRY.register()
def internimage_xl_22kto1k_384(
    pretrained, weight_path, features_only=True, out_indices=(0, 1, 2, 3), **kwargs
):
    model = create_model(
        "internimage_xl_22kto1k_384",
        pretrained=pretrained,
        weight_path=weight_path,
        features_only=features_only,
        out_indices=out_indices,
        **kwargs,
    )
    return model


@BACKBONE_REGISTRY.register()
def internimage_h_22kto1k_384(
    pretrained, weight_path, features_only=True, out_indices=(0, 1, 2, 3), **kwargs
):
    model = create_model(
        "internimage_h_22kto1k_384",
        pretrained=pretrained,
        weight_path=weight_path,
        features_only=features_only,
        out_indices=out_indices,
        **kwargs,
    )
    return model


@BACKBONE_REGISTRY.register()
def internimage_h_22kto1k_640(
    pretrained, weight_path, features_only=True, out_indices=(0, 1, 2, 3), **kwargs
):
    model = create_model(
        "internimage_h_22kto1k_640",
        pretrained=pretrained,
        weight_path=weight_path,
        features_only=features_only,
        out_indices=out_indices,
        **kwargs,
    )
    return model


@BACKBONE_REGISTRY.register()
def internimage_g_22kto1k_512(
    pretrained, weight_path, features_only=True, out_indices=(0, 1, 2, 3), **kwargs
):
    model = create_model(
        "internimage_g_22kto1k_512",
        pretrained=pretrained,
        weight_path=weight_path,
        features_only=features_only,
        out_indices=out_indices,
        **kwargs,
    )
    return model
