import src.optimizer as mod

from ..utils import import_submodules
from .build import OPTIMIZER_REGISTRY, build_optimizer

import_submodules(mod)
