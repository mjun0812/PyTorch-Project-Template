import src.models.losses as mod

from ...utils import import_submodules
from .base import BaseLoss
from .build import LOSS_REGISTRY, build_loss

import_submodules(mod)
