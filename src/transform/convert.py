import numpy as np
import torch

from ..dataloaders import DatasetOutput
from .base import BaseTransform
from .build import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register()
class ToTensor(BaseTransform):
    def __call__(self, data: DatasetOutput) -> DatasetOutput:
        for key in data:
            if isinstance(data[key], list) and not isinstance(data[key][0], str):
                data[key] = torch.tensor(data[key])
            elif isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key])
        return data
