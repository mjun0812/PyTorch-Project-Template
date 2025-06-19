import torchvision.transforms.v2 as T

from ..config import TransformConfig
from ..utils import Registry
from .base import BaseTransform

TRANSFORM_REGISTRY = Registry("TRANSFORM")
BATCHED_TRANSFORM_REGISTRY = Registry("BATCHED_TRANSFORM")


def build_transform(cfg: TransformConfig) -> BaseTransform:
    args = cfg.args or {}
    transform = TRANSFORM_REGISTRY.get(cfg.class_name)(**args)
    return transform


def build_batch_transform(cfg: TransformConfig) -> BaseTransform:
    args = cfg.args or {}
    batch_transform = BATCHED_TRANSFORM_REGISTRY.get(cfg.class_name)(**args)
    return batch_transform


def build_transforms(cfg: list[TransformConfig]) -> T.Compose:
    transforms = []
    for cfg_transform in cfg:
        transforms.append(build_transform(cfg_transform))
    return T.Compose(transforms)


def build_batched_transform(cfg: list[TransformConfig]) -> T.Compose:
    batched_transforms = []
    for cfg_transform in cfg:
        batched_transforms.append(build_batch_transform(cfg_transform))
    return T.Compose(batched_transforms)
