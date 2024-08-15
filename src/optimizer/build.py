import torch
from loguru import logger
from torch import optim

from ..config import ConfigManager, ExperimentConfig, OptimizerGroupConfig
from ..utils import check_model_parallel, get_world_size
from .lion import Lion


def adjust_learning_rate(base_lr: float, batch_size: int):
    world_size = get_world_size()
    return base_lr * batch_size * world_size


def build_optimizer(cfg: ExperimentConfig, model: torch.nn.Module) -> optim.Optimizer:
    optimizer_cls = cfg.optimizer.optimizer
    lr = cfg.optimizer.lr
    if cfg.adjust_lr:
        lr = adjust_learning_rate(lr, cfg.batch)

    target_model = model
    if check_model_parallel(model):
        target_model = model.module

    if cfg.optimizer.group is not None:
        parameters = get_param_group(target_model, cfg.optimizer.group, lr)
    else:
        parameters = target_model.parameters()

    args = cfg.optimizer.get("args")
    if args is not None:
        args = ConfigManager.to_object(args.copy())
    else:
        args = {}

    if optimizer_cls == "AdamW":
        optimizer = optim.AdamW(parameters, lr=lr, **args)
    elif optimizer_cls == "Adam":
        optimizer = optim.Adam(parameters, lr=lr, **args)
    elif optimizer_cls == "NesterovMomentum":
        optimizer = optim.SGD(parameters, lr=lr, **args)
    elif optimizer_cls == "Momentum":
        optimizer = optim.SGD(parameters, lr=lr, **args)
    elif optimizer_cls == "SGD":
        optimizer = optim.SGD(parameters, lr=lr, **args)
    elif optimizer_cls == "Lion":
        optimizer = Lion(parameters, lr=lr, **args)

    if cfg.optimizer.checkpoint is not None:
        optimizer.load_state_dict(
            torch.load(cfg.optimizer.checkpoint, map_location="cpu", weights_only=True)
        )

    logger.info(f"Optimizer: {cfg.optimizer.optimizer}")
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


def get_param_group(
    model: torch.nn.Module, cfg: list[OptimizerGroupConfig], base_lr: float
) -> list:
    optimizer_dict = [{"params": [], "lr": base_lr}]
    keys = []
    for info in cfg:
        optimizer_dict.append({"params": [], "lr": base_lr / info.divide})
        keys.append(info.name)

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
