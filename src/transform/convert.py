import numpy as np
import torch

from ..dataloaders import DatasetOutput
from .base import BaseTransform
from .build import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register()
class ToTensor(BaseTransform):
    """Transform that converts data to PyTorch tensors.

    Converts NumPy arrays and numeric lists to PyTorch tensors while
    preserving string data unchanged. This is commonly used as the final
    step in preprocessing pipelines to prepare data for model input.

    The transformation handles:
    - NumPy arrays: Converted using torch.from_numpy()
    - Numeric lists: Converted using torch.tensor()
    - String lists: Left unchanged
    - Other data types: Left unchanged
    """

    def forward(self, data: DatasetOutput) -> DatasetOutput:
        """Convert applicable data types to PyTorch tensors.

        Args:
            data: Input data dictionary containing various data types.

        Returns:
            Data dictionary with numeric arrays/lists converted to tensors.
        """
        for key in data:
            if isinstance(data[key], list) and not isinstance(data[key][0], str):
                data[key] = torch.tensor(data[key])
            elif isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key])
        return data
