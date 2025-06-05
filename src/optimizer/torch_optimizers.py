from torch import optim

from .build import OPTIMIZER_REGISTRY

OPTIMIZER_REGISTRY.register(optim.AdamW, name="AdamW")
OPTIMIZER_REGISTRY.register(optim.Adam, name="Adam")
OPTIMIZER_REGISTRY.register(optim.SGD, name="NesterovMomentum")
OPTIMIZER_REGISTRY.register(optim.SGD, name="Momentum")
OPTIMIZER_REGISTRY.register(optim.SGD, name="SGD")
