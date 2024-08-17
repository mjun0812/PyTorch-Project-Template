import torch
import torch.nn as nn
from torch.nn import functional as F

from ..alias import ModelOutput, PhaseStr
from ..config import ExperimentConfig
from .base import BaseModel
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class DummyModel(BaseModel):
    def __init__(self, cfg: ExperimentConfig, phase: PhaseStr = "train"):
        super().__init__(cfg, phase)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def train_forward(self, data: dict) -> ModelOutput:
        output = self.test_forward(data)
        loss = self.loss(data, output["preds"])
        return ModelOutput(losses=loss, preds=output["preds"])

    def val_forward(self, data: dict) -> ModelOutput:
        return self.train_forward(data)

    def test_forward(self, data: dict) -> ModelOutput:
        x = self.conv1(data["data"])
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return ModelOutput(preds={"pred": output})
