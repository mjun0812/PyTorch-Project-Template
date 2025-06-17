from ...config import LossConfig
from ...utils import Registry
from .base import BaseLoss

LOSS_REGISTRY = Registry("LOSS")


def build_loss(cfg: LossConfig) -> BaseLoss:
    loss_name = cfg.class_name
    if cfg.args is None:
        loss = LOSS_REGISTRY.get(loss_name)()
    else:
        loss = LOSS_REGISTRY.get(loss_name)(**cfg.args)
    return loss
