import torch
from omegaconf import OmegaConf

from ..config import TransformConfig
from ..dataloaders import DatasetOutput


class BatchedTransformCompose:
    def __init__(self, cfg: list[TransformConfig]) -> None:
        from .build import BATCHED_TRANSFORM_REGISTRY

        self.transforms = []

        for c in cfg:
            if c.args is None:
                transform = BATCHED_TRANSFORM_REGISTRY.get(c.name)()
            else:
                args = OmegaConf.to_object(c.args)
                transform = BATCHED_TRANSFORM_REGISTRY.get(c.name)(**args)
            self.transforms.append(transform)

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
