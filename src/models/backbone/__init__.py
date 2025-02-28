# flake8: noqa

from .build import BACKBONE_REGISTRY, BackboneConfig, build_backbone, get_available_backbones
from .internimage import (
    internimage_b_1k_224,
    internimage_g_22kto1k_512,
    internimage_h_22kto1k_384,
    internimage_h_22kto1k_640,
    internimage_l_22kto1k_384,
    internimage_s_1k_224,
    internimage_t_1k_224,
    internimage_xl_22kto1k_384,
)
from .swin_transformer import (
    swin_base_patch4_window7_224,
    swin_base_patch4_window7_224_22k,
    swin_base_patch4_window7_224_22kto1k,
    swin_base_patch4_window12_384,
    swin_base_patch4_window12_384_22k,
    swin_base_patch4_window12_384_22kto1k,
    swin_large_patch4_window7_224_22k,
    swin_large_patch4_window7_224_22kto1k,
    swin_large_patch4_window12_384_22k,
    swin_large_patch4_window12_384_22kto1k,
    swin_small_patch4_window7_224,
    swin_small_patch4_window7_224_22k,
    swin_small_patch4_window7_224_22kto1k,
    swin_tiny_patch4_window7_224,
    swin_tiny_patch4_window7_224_22k,
    swin_tiny_patch4_window7_224_22kto1k,
)
