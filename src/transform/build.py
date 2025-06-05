import torchvision.transforms.v2 as T

from ..config import TransformConfig
from ..utils import Registry
from .compose import BatchedTransformCompose

TRANSFORM_REGISTRY = Registry("TRANSFORM")
BATCHED_TRANSFORM_REGISTRY = Registry("BATCHED_TRANSFORM")


def build_transform(cfg: TransformConfig) -> T.Transform:
    if cfg.args is None:
        transform = TRANSFORM_REGISTRY.get(cfg.class_name)()
    else:
        transform = TRANSFORM_REGISTRY.get(cfg.class_name)(**cfg.args)
    return transform


def build_transforms(cfg: list[TransformConfig]) -> T.Compose:
    transforms = []
    for cfg_transform in cfg:
        transforms.append(build_transform(cfg_transform))
    return T.Compose(transforms)


def build_batched_transform(cfg: list[TransformConfig]) -> BatchedTransformCompose:
    batched_transforms = []
    for cfg_batched_transform in cfg:
        batched_transforms.append(build_transform(cfg_batched_transform))
    return BatchedTransformCompose(batched_transforms)
