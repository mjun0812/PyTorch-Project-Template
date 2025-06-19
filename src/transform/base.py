from abc import abstractmethod

from torch import nn

from ..dataloaders import DatasetOutput


class BaseTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, data: DatasetOutput) -> DatasetOutput:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = [f"{k}={v}" for k, v in self.__dict__.items()]
        if len(args) == 0:
            return self.__class__.__name__ + "()"
        return self.__class__.__name__ + "(" + ", ".join(args) + ")"
