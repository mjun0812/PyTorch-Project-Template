import logging

import torch.optim as optim
from kunai.torch_utils import check_model_parallel

from ..utils import adjust_learning_rate
from .lion import Lion

logger = logging.getLogger()


def build_optimizer(cfg, model):
    optimizer_name = cfg.OPTIMIZER.NAME
    lr = cfg.OPTIMIZER.LR
    if cfg.ADJUST_LR:
        lr = adjust_learning_rate(lr, cfg.BATCH)

    target_model = model
    if check_model_parallel(model):
        target_model = model.module

    if hasattr(target_model, "optimizer_dict"):
        parameters = target_model.optimizer_dict
    elif "WEIGHT_DECAY" in cfg.OPTIMIZER:
        parameters = param_groups_weight_decay(
            target_model, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
        )
    else:
        parameters = target_model.parameters()

    if optimizer_name == "AdamW":
        args = {"weight_decay": cfg.OPTIMIZER.get("WEIGHT_DECAY", 1e-2)}
        optimizer = optim.AdamW(parameters, lr=lr, **args)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(parameters, lr=lr)
    elif optimizer_name == "NesterovMomentum":
        optimizer = optim.SGD(
            parameters,
            lr=lr,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            nesterov=True,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif optimizer_name == "Momentum":
        optimizer = optim.SGD(
            parameters,
            lr=lr,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            nesterov=False,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(parameters, lr=lr)
    elif optimizer_name == "Lion":
        optimizer = Lion(parameters, lr=lr, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    logger.info(f"Optimizer: {optimizer_name}")
    logger.info(f"Optimizer Group: {optimizer}")
    logger.info(f"Learning Rate: {lr}")
    return optimizer


def param_groups_weight_decay(model, weight_decay=1e-5, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
