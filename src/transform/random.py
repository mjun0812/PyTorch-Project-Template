import random

import numpy as np
import torch

from ..config import TransformConfig
from .build import TRANSFORM_REGISTRY, get_transform
from .transform import BaseTransform

USE_V2 = False
try:
    USE_V2 = True
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as TF
except ImportError:
    print("ImportError: torchvision.transforms.v2")
    import torchvision.transforms.functional as TF
    from torchvision import transforms as T


@TRANSFORM_REGISTRY.register()
class RandomSelect(BaseTransform):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(
        self, transforms1: list[TransformConfig], transforms2: list[TransformConfig], p=0.5
    ):
        self.transforms1 = T.Compose([get_transform(t) for t in transforms1])
        self.transforms2 = T.Compose([get_transform(t) for t in transforms2])
        self.p = p

    def __call__(self, data: dict) -> dict:
        if random.random() < self.p:
            return self.transforms1(data)
        return self.transforms2(data)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += f"{self.transforms1}, "
        format_string += f"{self.transforms2})"
        return format_string


@TRANSFORM_REGISTRY.register()
class ToTensor(BaseTransform):
    def __call__(self, data: dict) -> dict:
        for key in data:
            if isinstance(data[key], list):
                data[key] = torch.tensor(data[key])
            elif isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key])
        return data
