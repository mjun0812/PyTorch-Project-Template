# flake8: noqa
import src.models.losses as mod

from ...utils import import_submodules
from .build import LOSS_REGISTRY, build_loss
from .base import BaseLoss

import_submodules(mod)
