from timm.optim import (
    MADGRAD,
    SGD,
    SGDP,
    SGDW,
    AdaBelief,
    Adafactor,
    AdafactorBigVision,
    Adahessian,
    AdamP,
    AdamWLegacy,
    Adan,
    Adopt,
    Lamb,
    LaProp,
    Lars,
    Lion,
    Lookahead,
    Mars,
    NAdamLegacy,
    NAdamW,
    NvNovoGrad,
    RAdamLegacy,
    RMSpropTF,
)

from .build import OPTIMIZER_REGISTRY

OPTIMIZER_REGISTRY.register(MADGRAD, name="timm_MADGRAD")
OPTIMIZER_REGISTRY.register(SGD, name="timm_SGD")
OPTIMIZER_REGISTRY.register(SGDP, name="timm_SGDP")
OPTIMIZER_REGISTRY.register(SGDW, name="timm_SGDW")
OPTIMIZER_REGISTRY.register(AdaBelief, name="timm_AdaBelief")
OPTIMIZER_REGISTRY.register(Adafactor, name="timm_Adafactor")
OPTIMIZER_REGISTRY.register(AdafactorBigVision, name="timm_AdafactorBigVision")
OPTIMIZER_REGISTRY.register(Adahessian, name="timm_Adahessian")
OPTIMIZER_REGISTRY.register(AdamP, name="timm_AdamP")
OPTIMIZER_REGISTRY.register(AdamWLegacy, name="timm_AdamWLegacy")
OPTIMIZER_REGISTRY.register(Adan, name="timm_Adan")
OPTIMIZER_REGISTRY.register(Adopt, name="timm_Adopt")
OPTIMIZER_REGISTRY.register(Lamb, name="timm_Lamb")
OPTIMIZER_REGISTRY.register(LaProp, name="timm_LaProp")
OPTIMIZER_REGISTRY.register(Lars, name="timm_Lars")
OPTIMIZER_REGISTRY.register(Lion, name="timm_Lion")
OPTIMIZER_REGISTRY.register(Lookahead, name="timm_Lookahead")
OPTIMIZER_REGISTRY.register(Mars, name="timm_Mars")
OPTIMIZER_REGISTRY.register(NAdamLegacy, name="timm_NAdamLegacy")
OPTIMIZER_REGISTRY.register(NAdamW, name="timm_NAdamW")
OPTIMIZER_REGISTRY.register(NvNovoGrad, name="timm_NvNovoGrad")
OPTIMIZER_REGISTRY.register(RAdamLegacy, name="timm_RAdamLegacy")
OPTIMIZER_REGISTRY.register(RMSpropTF, name="timm_RMSpropTF")
