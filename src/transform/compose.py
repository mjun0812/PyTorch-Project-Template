import torch
import torch.nn as nn

from ..dataloaders import DatasetOutput


class BatchedTransformCompose:
    def __init__(self, transforms: list[nn.Module]) -> None:
        self.transforms = transforms

    def to(self, device: torch.device) -> None:
        for t in self.transforms:
            if isinstance(t, torch.nn.Module):
                t.to(device)

    def __call__(self, data: DatasetOutput) -> DatasetOutput:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
