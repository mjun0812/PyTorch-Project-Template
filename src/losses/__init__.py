# flake8: noqa
import src.losses as mod

from ..utils import import_submodules
from .build import LOSS_REGISTRY, build_loss, BaseLoss, LossOutput
from .loss import DummyLoss

import_submodules(mod)
