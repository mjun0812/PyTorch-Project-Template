from typing import Required, TypedDict

import torch.nn as nn
from torch import Tensor

from ..config import ExperimentConfig
from ..utils import Registry

LOSS_REGISTRY = Registry("LOSS")


class LossOutput(TypedDict, total=False):
    total_loss: Required[Tensor]


class BaseLoss(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, targets: dict, preds: dict) -> LossOutput:
        raise NotImplementedError


def build_loss(cfg: ExperimentConfig) -> BaseLoss:
    loss_name = cfg.model.loss.loss
    loss = LOSS_REGISTRY.get(loss_name)(cfg)
    return loss
