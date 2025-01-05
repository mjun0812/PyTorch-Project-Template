import torch.optim as optim

from .build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class AdamW(optim.AdamW):
    pass


@OPTIMIZER_REGISTRY.register()
class Adam(optim.Adam):
    pass


@OPTIMIZER_REGISTRY.register()
class NesterovMomentum(optim.SGD):
    pass


@OPTIMIZER_REGISTRY.register()
class Momentum(optim.SGD):
    pass


@OPTIMIZER_REGISTRY.register()
class SGD(optim.SGD):
    pass
