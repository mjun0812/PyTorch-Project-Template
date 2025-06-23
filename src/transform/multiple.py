import random

import torchvision.transforms.v2 as T

from ..config import TransformConfig
from ..dataloaders import DatasetOutput
from .base import BaseTransform
from .build import TRANSFORM_REGISTRY, build_transform


@TRANSFORM_REGISTRY.register()
class RandomSelect(BaseTransform):
    """Randomly selects between two sets of transforms.

    This transform provides data augmentation by randomly choosing between
    two different transformation pipelines with specified probabilities.
    Useful for creating diverse augmentation strategies during training.

    Args:
        transforms1: List of transform configurations for the first pipeline.
        transforms2: List of transform configurations for the second pipeline.
        p: Probability of selecting transforms1 (1-p for transforms2).

    Attributes:
        transforms1: Composed transformation pipeline 1.
        transforms2: Composed transformation pipeline 2.
        p: Selection probability for transforms1.
    """

    def __init__(
        self,
        transforms1: list[TransformConfig],
        transforms2: list[TransformConfig],
        p: float = 0.5,
    ) -> None:
        """Initialize the random selection transform.

        Args:
            transforms1: List of transform configurations for pipeline 1.
            transforms2: List of transform configurations for pipeline 2.
            p: Probability of selecting transforms1. Defaults to 0.5.
        """
        super().__init__()
        self.transforms1 = T.Compose([build_transform(TransformConfig(**t)) for t in transforms1])
        self.transforms2 = T.Compose([build_transform(TransformConfig(**t)) for t in transforms2])
        self.p = p

    def forward(self, data: DatasetOutput) -> DatasetOutput:
        """Apply randomly selected transformation pipeline.

        Args:
            data: Input data dictionary.

        Returns:
            Transformed data using either transforms1 or transforms2.
        """
        if random.random() < self.p:
            return self.transforms1(data)
        return self.transforms2(data)

    def __repr__(self) -> str:
        """Return string representation of the transform.

        Returns:
            String representation showing both transform pipelines.
        """
        format_string = self.__class__.__name__ + "("
        format_string += f"{self.transforms1}, "
        format_string += f"{self.transforms2})"
        return format_string
