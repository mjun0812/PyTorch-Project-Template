# flake8: noqa
import src.losses as mod

from ..utils import import_submodules
from .build import LOSS_REGISTRY, build_loss
from .base import BaseLoss
from .loss import DummyLoss

import_submodules(mod)
