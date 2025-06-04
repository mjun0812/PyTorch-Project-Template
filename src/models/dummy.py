from dataclasses import dataclass

from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from ..dataloaders.types import DatasetOutput
from ..types import PhaseStr
from .base import BaseModel
from .build import MODEL_REGISTRY
from .types import ModelOutput


@dataclass
class DummyModelConfig:
    input_channels: int = 1


@MODEL_REGISTRY.register()
class DummyModel(BaseModel):
    def __init__(self, cfg: ModelConfig, phase: PhaseStr = "train"):
        super().__init__(cfg, phase)
        self.cfg_model = DummyModelConfig(**self.cfg.args)
        self.fc = nn.Linear(8, 4)

    def train_forward(self, data: DatasetOutput) -> ModelOutput:
        output = self.test_forward(data)
        loss = self.loss(data, output["preds"])
        return ModelOutput(losses=loss, preds=output["preds"])

    def val_forward(self, data: DatasetOutput) -> ModelOutput:
        return self.train_forward(data)

    def test_forward(self, data: DatasetOutput) -> ModelOutput:
        x = data["data"]
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return ModelOutput(preds={"pred": output})
