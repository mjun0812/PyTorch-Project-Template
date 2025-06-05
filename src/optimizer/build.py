from typing import Any

import torch
from timm.optim import create_optimizer_v2
from torch import optim

from ..config import ConfigManager, OptimizerConfig, OptimizerGroupConfig
from ..utils import Registry, is_model_parallel

OPTIMIZER_REGISTRY = Registry("OPTIMIZER")


def build_optimizer(cfg: OptimizerConfig, model: torch.nn.Module) -> optim.Optimizer:
    optimizer_cls_name = cfg.class_name
    lr = cfg.lr

    target_model = model
    if is_model_parallel(model):
        target_model = model.module

    if cfg.group is not None:
        parameters = get_param_group(target_model, cfg.group, lr)
    else:
        parameters = target_model.parameters()

    args = cfg.get("args")
    if args is not None:
        args = ConfigManager.to_object(args.copy())
    else:
        args = {}

    if "timm" not in optimizer_cls_name:
        optimizer = OPTIMIZER_REGISTRY.get(optimizer_cls_name)(parameters, lr=lr, **args)
    else:
        optimizer_cls_name = optimizer_cls_name.replace("_timm", "")
        optimizer = create_optimizer_v2(target_model, opt=optimizer_cls_name, lr=lr, **args)

    if cfg.checkpoint is not None:
        optimizer.load_state_dict(
            torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        )

    return optimizer


def param_groups_weight_decay(
    model: torch.nn.Module,
    weight_decay: float = 1e-5,
    no_weight_decay_list: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
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
