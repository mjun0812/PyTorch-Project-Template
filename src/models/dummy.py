from dataclasses import dataclass

from torch import nn
from torch.nn import functional as F

from ..dataloaders import DatasetOutput
from ..types import PhaseStr
from .base import BaseModel
from .build import MODEL_REGISTRY
from .types import ModelOutput


@dataclass
class DummyModelConfig:
    """Configuration for the dummy model.

    Attributes:
        input_channels: Number of input channels for the model.
    """

    input_channels: int = 1


@MODEL_REGISTRY.register()
class DummyModel(BaseModel):
    """Dummy model for testing and demonstration purposes.

    A simple neural network with a single linear layer that transforms
    8-dimensional input to 4-dimensional output with log softmax activation.
    """

    def __init__(
        self, cfg: dict | None, cfg_loss: dict | None = None, phase: PhaseStr = "train"
    ) -> None:
        """Initialize the dummy model.

        Args:
            cfg: Model configuration dictionary.
            cfg_loss: Loss function configuration dictionary.
            phase: Training phase (train, val, test).
        """
        super().__init__(cfg, cfg_loss, phase)
        self.cfg = DummyModelConfig(**(self.cfg or {}))
        self.fc = nn.Linear(8, 4)

    def train_forward(self, data: DatasetOutput) -> ModelOutput:
        """Forward pass for training phase.

        Args:
            data: Input data from the dataset.

        Returns:
            Model output containing predictions and computed losses.
        """
        output = self.test_forward(data)
        loss = self.loss(data, output["preds"])
        return ModelOutput(losses=loss, preds=output["preds"])

    def val_forward(self, data: DatasetOutput) -> ModelOutput:
        """Forward pass for validation phase.

        Args:
            data: Input data from the dataset.

        Returns:
            Model output containing predictions and computed losses.
        """
        return self.train_forward(data)

    def test_forward(self, data: DatasetOutput) -> ModelOutput:
        """Forward pass for testing phase.

        Args:
            data: Input data from the dataset.

        Returns:
            Model output containing predictions without losses.
        """
        x = data["data"]
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return ModelOutput(preds={"pred": output})
