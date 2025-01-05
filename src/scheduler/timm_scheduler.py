from timm.scheduler import CosineLRScheduler

from .build import SCHEDULER_REGISTRY

SCHEDULER_REGISTRY.register(CosineLRScheduler)
