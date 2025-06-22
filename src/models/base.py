from abc import abstractmethod

from torch import nn

from ..dataloaders import DatasetOutput
from ..types import PhaseStr
from .losses import build_loss
from .types import ModelOutput


class BaseModel(nn.Module):
    def __init__(
        self, cfg: dict | None, cfg_loss: dict | None = None, phase: PhaseStr = "train"
    ) -> None:
        """
        Base model class.

        Args:
            cfg: Configuration dictionary.
            cfg_loss: Configuration dictionary for loss.
            phase: Phase of the model. (train, val, test)
        """
        super().__init__()
        self.cfg = cfg
        self.cfg_loss = cfg_loss
        self.phase = phase

        self.loss = None
        if self.phase in ["train", "val"] and self.cfg_loss is not None:
            self.loss = build_loss(self.cfg_loss)

    @abstractmethod
    def train_forward(self, data: DatasetOutput) -> ModelOutput:
        raise NotImplementedError

    @abstractmethod
    def val_forward(self, data: DatasetOutput) -> ModelOutput:
        raise NotImplementedError

    @abstractmethod
    def test_forward(self, data: DatasetOutput) -> ModelOutput:
        raise NotImplementedError

    def forward(self, data: DatasetOutput) -> ModelOutput:
        if self.phase == "train":
            return self.train_forward(data)
        elif self.phase == "val":
            return self.val_forward(data)
        else:
            return self.test_forward(data)
