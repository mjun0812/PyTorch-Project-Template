import src.scheduler as mod

from ..utils import import_submodules
from .build import SCHEDULER_REGISTRY, build_lr_scheduler, get_lr_scheduler

import_submodules(mod)
