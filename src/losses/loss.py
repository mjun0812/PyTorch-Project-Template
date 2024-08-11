from ..config import ExperimentConfig
from .build import LOSS_REGISTRY, BaseLoss, LossOutput


@LOSS_REGISTRY.register()
class DummyLoss(BaseLoss):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)

    def forward(self, targets: dict, preds: dict) -> LossOutput:
        return LossOutput(total_loss=preds["pred"].sum())
