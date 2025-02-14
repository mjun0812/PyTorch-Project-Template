from ...config import ExperimentConfig
from ...utils import Registry
from .base import BaseLoss

LOSS_REGISTRY = Registry("LOSS")


def build_loss(cfg: ExperimentConfig) -> BaseLoss:
    loss_name = cfg.model.loss.loss
    loss = LOSS_REGISTRY.get(loss_name)(cfg)
    return loss
