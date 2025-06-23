from abc import abstractmethod

from torch import nn

from ..dataloaders import DatasetOutput
from ..types import PhaseStr
from .losses import build_loss
from .types import ModelOutput


class BaseModel(nn.Module):
    """Abstract base class for all models in the framework.

    Provides a common interface for models with phase-aware forward passes
    and automatic loss function integration.

    Attributes:
        cfg: Model configuration dictionary.
        cfg_loss: Loss function configuration dictionary.
        phase: Current training phase (train, val, test).
        loss: Loss function instance (only available for train/val phases).
    """

    def __init__(
        self, cfg: dict | None, cfg_loss: dict | None = None, phase: PhaseStr = "train"
    ) -> None:
        """Initialize the base model.

        Args:
            cfg: Model configuration dictionary.
            cfg_loss: Loss function configuration dictionary.
            phase: Training phase (train, val, test).
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
        """Forward pass for training phase.

        Args:
            data: Input data from the dataset.

        Returns:
            Model output containing predictions and losses.
        """
        raise NotImplementedError

    @abstractmethod
    def val_forward(self, data: DatasetOutput) -> ModelOutput:
        """Forward pass for validation phase.

        Args:
            data: Input data from the dataset.

        Returns:
            Model output containing predictions and optionally losses.
        """
        raise NotImplementedError

    @abstractmethod
    def test_forward(self, data: DatasetOutput) -> ModelOutput:
        """Forward pass for testing phase.

        Args:
            data: Input data from the dataset.

        Returns:
            Model output containing predictions.
        """
        raise NotImplementedError

    def forward(self, data: DatasetOutput) -> ModelOutput:
        """Phase-aware forward pass.

        Automatically routes to the appropriate forward method based on the
        current phase (train_forward, val_forward, or test_forward).

        Args:
            data: Input data from the dataset.

        Returns:
            Model output appropriate for the current phase.
        """
        if self.phase == "train":
            return self.train_forward(data)
        elif self.phase == "val":
            return self.val_forward(data)
        else:
            return self.test_forward(data)
