import torchvision.transforms.v2 as T

from ..config import TransformConfig
from ..utils import Registry
from .base import BaseTransform
from .batch_compose import BatchedTransformCompose

TRANSFORM_REGISTRY = Registry("TRANSFORM")
"""Registry for data transformation classes."""

BATCHED_TRANSFORM_REGISTRY = Registry("BATCHED_TRANSFORM")
"""Registry for batch-level transformation classes."""


def build_transform(cfg: TransformConfig) -> BaseTransform:
    """Build a single data transformation from configuration.

    Args:
        cfg: Transform configuration containing class name and arguments.

    Returns:
        Instantiated transformation object.
    """
    args = cfg.args or {}
    transform = TRANSFORM_REGISTRY.get(cfg.class_name)(**args)
    return transform


def build_batch_transform(cfg: TransformConfig) -> BaseTransform:
    """Build a single batch-level transformation from configuration.

    Args:
        cfg: Transform configuration containing class name and arguments.

    Returns:
        Instantiated batch transformation object.
    """
    args = cfg.args or {}
    batch_transform = BATCHED_TRANSFORM_REGISTRY.get(cfg.class_name)(**args)
    return batch_transform


def build_transforms(cfg: list[TransformConfig]) -> T.Compose:
    """Build a composition of data transformations.

    Args:
        cfg: List of transform configurations to compose.

    Returns:
        Composed transformation pipeline.
    """
    transforms = []
    for cfg_transform in cfg:
        transforms.append(build_transform(cfg_transform))
    return T.Compose(transforms)


def build_batched_transform(cfg: list[TransformConfig]) -> BatchedTransformCompose:
    """Build a composition of batch-level transformations.

    Args:
        cfg: List of batch transform configurations to compose.

    Returns:
        Composed batch transformation pipeline.
    """
    batched_transforms = []
    for cfg_transform in cfg:
        batched_transforms.append(build_batch_transform(cfg_transform))
    return BatchedTransformCompose(batched_transforms)
