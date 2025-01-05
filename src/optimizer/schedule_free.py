from schedulefree import AdamWScheduleFree, RAdamScheduleFree

from .build import OPTIMIZER_REGISTRY

OPTIMIZER_REGISTRY.register(AdamWScheduleFree)
OPTIMIZER_REGISTRY.register(RAdamScheduleFree)
