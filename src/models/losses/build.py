from ...config import LossConfig
from ...utils import Registry
from .base import BaseLoss

LOSS_REGISTRY = Registry("LOSS")
"""Registry for loss function classes."""


def build_loss(cfg: LossConfig) -> BaseLoss:
    """Build a loss function instance from configuration.

    Args:
        cfg: Loss configuration containing class name and arguments.

    Returns:
        Instantiated loss function.
    """
    loss = LOSS_REGISTRY.get(cfg.class_name)(cfg.args)
    return loss
