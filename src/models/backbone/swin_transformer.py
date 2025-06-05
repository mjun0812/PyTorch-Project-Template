# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

from typing import Any

import torch

from .build import BACKBONE_REGISTRY
from .swin_transformer_impl import SwinTransformer


# ####### Tiny Model #######
@BACKBONE_REGISTRY.register()
def swin_tiny_patch4_window7_224_22k(pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any) -> Any:
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.1,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_tiny_patch4_window7_224_22k.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_tiny_patch4_window7_224_22kto1k(
    pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any
) -> Any:
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.1,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = (
                "./model_zoo/swin-transformer/swin_tiny_patch4_window7_224_22kto1k_finetune.pth"
            )
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_tiny_patch4_window7_224(pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any) -> Any:
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.1,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_tiny_patch4_window7_224.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


# ####### Small Model #######
@BACKBONE_REGISTRY.register()
def swin_small_patch4_window7_224_22k(pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any) -> Any:
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_small_patch4_window7_224_22k.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_small_patch4_window7_224_22kto1k(
    pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any
) -> Any:
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = (
                "./model_zoo/swin-transformer/swin_small_patch4_window7_224_22kto1k_finetune.pth"
            )
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_small_patch4_window7_224(pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any) -> Any:
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_small_patch4_window7_224.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


# ####### Base Model #######
@BACKBONE_REGISTRY.register()
def swin_base_patch4_window7_224_22k(pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any) -> Any:
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_base_patch4_window7_224_22k.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_base_patch4_window7_224_22kto1k(
    pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any
) -> Any:
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_base_patch4_window7_224_22kto1k.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_base_patch4_window7_224(pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any) -> Any:
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_base_patch4_window7_224.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_base_patch4_window12_384(pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any) -> Any:
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_base_patch4_window12_384.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_base_patch4_window12_384_22k(pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any) -> Any:
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_base_patch4_window12_384_22k.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_base_patch4_window12_384_22kto1k(
    pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any
) -> Any:
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_base_patch4_window12_384_22kto1k.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


# ####### Large Model #######
@BACKBONE_REGISTRY.register()
def swin_large_patch4_window7_224_22k(pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any) -> Any:
    model = SwinTransformer(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_large_patch4_window7_224_22k.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_large_patch4_window7_224_22kto1k(
    pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any
) -> Any:
    model = SwinTransformer(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_large_patch4_window7_224_22kto1k.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_large_patch4_window12_384_22k(pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any) -> Any:
    model = SwinTransformer(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_large_patch4_window12_384_22k.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model


@BACKBONE_REGISTRY.register()
def swin_large_patch4_window12_384_22kto1k(
    pretrained: bool, weight_path: str | None, out_indices: tuple[int, ...] = (0, 1, 2, 3), **kwargs: Any
) -> Any:
    model = SwinTransformer(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        **kwargs,
    )
    if pretrained:
        if not weight_path:
            weight_path = "./model_zoo/swin-transformer/swin_large_patch4_window12_384_22kto1k.pth"
        checkpoint = torch.load(weight_path)["model"]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    return model
