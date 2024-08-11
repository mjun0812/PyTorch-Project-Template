import torch.nn as nn

from ..alias import PhaseStr
from ..config import ExperimentConfig
from .build import MODEL_REGISTRY, BaseModel, ModelOutput


@MODEL_REGISTRY.register()
class DummyModel(BaseModel):
    def __init__(self, cfg: ExperimentConfig, phase: PhaseStr = "train"):
        super().__init__(cfg, phase)
        self.model = nn.Linear(256, 10)

    def train_forward(self, data: dict) -> ModelOutput:
        loss = self.loss(data, {"pred": self.model(data["image"])})
        return ModelOutput(losses=loss)

    def val_forward(self, data: dict) -> ModelOutput:
        x = self.model(data["image"])
        loss = self.loss(data, {"pred": x})
        return ModelOutput(losses=loss, preds={"pred": x})

    def test_forward(self, data: dict) -> ModelOutput:
        x = self.model(data["image"])
        return ModelOutput(preds={"pred": x})
