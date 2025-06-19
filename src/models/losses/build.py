from ...config import LossConfig
from ...utils import Registry
from .base import BaseLoss

LOSS_REGISTRY = Registry("LOSS")


def build_loss(cfg: LossConfig) -> BaseLoss:
    loss = LOSS_REGISTRY.get(cfg.class_name)(cfg.args)
    return loss
