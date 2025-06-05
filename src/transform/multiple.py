import random

import torchvision.transforms.v2 as T

from ..config import TransformConfig
from ..dataloaders import DatasetOutput
from .base import BaseTransform
from .build import TRANSFORM_REGISTRY, build_transform


@TRANSFORM_REGISTRY.register()
class RandomSelect(BaseTransform):
    """Randomly selects between transforms1 and transforms2.

    with probability p for transforms1 and (1 - p) for transforms2

    Args:
        transforms1 (list[TransformConfig]): List of transforms1.
        transforms2 (list[TransformConfig]): List of transforms2.
        p (float, optional): Probability of transforms1. Defaults to 0.5.
    """

    def __init__(
        self,
        transforms1: list[TransformConfig],
        transforms2: list[TransformConfig],
        p: float = 0.5,
    ) -> None:
        self.transforms1 = T.Compose([build_transform(TransformConfig(**t)) for t in transforms1])
        self.transforms2 = T.Compose([build_transform(TransformConfig(**t)) for t in transforms2])
        self.p = p

    def __call__(self, data: DatasetOutput) -> DatasetOutput:
        if random.random() < self.p:
            return self.transforms1(data)
        return self.transforms2(data)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"{self.transforms1}, "
        format_string += f"{self.transforms2})"
        return format_string
