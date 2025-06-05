from timm.scheduler import (
    CosineLRScheduler,
    MultiStepLRScheduler,
    PlateauLRScheduler,
    PolyLRScheduler,
    StepLRScheduler,
    TanhLRScheduler,
)

from .build import SCHEDULER_REGISTRY

SCHEDULER_REGISTRY.register(CosineLRScheduler, name="timm_CosineLRScheduler")
SCHEDULER_REGISTRY.register(TanhLRScheduler, name="timm_TanhLRScheduler")
SCHEDULER_REGISTRY.register(StepLRScheduler, name="timm_StepLRScheduler")
SCHEDULER_REGISTRY.register(MultiStepLRScheduler, name="timm_MultiStepLRScheduler")
SCHEDULER_REGISTRY.register(PolyLRScheduler, name="timm_PolyLRScheduler")
SCHEDULER_REGISTRY.register(PlateauLRScheduler, name="timm_PlateauLRScheduler")
