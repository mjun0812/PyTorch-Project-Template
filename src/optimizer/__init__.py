import src.optimizer as mod

from ..utils import import_submodules
from .build import OPTIMIZER_REGISTRY, adjust_learning_rate, build_optimizer

import_submodules(mod)
