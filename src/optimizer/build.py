import torch.optim as optim
from kunai.torch_utils import check_model_parallel

from ..utils import adjust_learning_rate
from .lion import Lion


def build_optimizer(cfg, model, logger=None):
    optimizer_name = cfg.OPTIMIZER.NAME
    lr = cfg.OPTIMIZER.LR
    if cfg.ADJUST_LR:
        lr = adjust_learning_rate(lr, cfg.BATCH)

    target_model = model
    if check_model_parallel(model):
        target_model = model.module

    if hasattr(target_model, "optimizer_dict"):
        parameters = target_model.optimizer_dict
    elif cfg.MODEL.get("LR"):
        parameters = get_param_group(target_model, cfg, lr)
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

    if logger is not None:
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


def get_param_group(model, cfg, base_lr):
    optimizer_dict = [{"params": [], "lr": base_lr}]
    keys = []
    for info in cfg.MODEL.LR:
        optimizer_dict.append({"params": [], "lr": base_lr / float(info["DIVIDE"])})
        keys.append(info["KEY"])

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        has_key = False
        for i, k in enumerate(keys):
            if k in n:
                optimizer_dict[i + 1]["params"].append(p)
                has_key = True
                break
        if not has_key:
            optimizer_dict[0]["params"].append(p)
    return optimizer_dict
