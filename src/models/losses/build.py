from ...config import LossConfig
from ...utils import Registry
from .base import BaseLoss

LOSS_REGISTRY = Registry("LOSS")


def build_loss(cfg: LossConfig) -> BaseLoss:
    loss_name = cfg.class_name
    loss = LOSS_REGISTRY.get(loss_name)(cfg)
    return loss
