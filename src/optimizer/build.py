from typing import Any

import torch
from torch import optim

from ..config import ConfigManager, OptimizerConfig, OptimizerGroupConfig
from ..utils import Registry, is_model_parallel

OPTIMIZER_REGISTRY = Registry("OPTIMIZER")
"""Registry for optimizer classes."""


def get_param_group(
    model: torch.nn.Module, cfg: list[OptimizerGroupConfig], base_lr: float
) -> list:
    """Create parameter groups with different learning rates.

    Organizes model parameters into groups based on parameter names,
    allowing different learning rates for different model components.

    Args:
        model: PyTorch model to extract parameters from.
        cfg: List of parameter group configurations.
        base_lr: Base learning rate for default parameters.

    Returns:
        List of parameter group dictionaries for optimizer.
    """
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


def param_groups_weight_decay(
    model: torch.nn.Module,
    weight_decay: float = 1e-5,
    no_weight_decay_list: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Create parameter groups with selective weight decay.

    Separates parameters into groups with and without weight decay,
    typically excluding bias terms and 1D parameters from weight decay.

    Args:
        model: PyTorch model to extract parameters from.
        weight_decay: Weight decay coefficient for applicable parameters.
        no_weight_decay_list: Parameter names to exclude from weight decay.

    Returns:
        List of parameter group dictionaries with weight decay settings.
    """
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


def build_optimizer(cfg: OptimizerConfig, model: torch.nn.Module) -> optim.Optimizer:
    """Build an optimizer instance from configuration.

    Args:
        cfg: Optimizer configuration containing class name, parameters, and checkpoints.
        model: PyTorch model to optimize.

    Returns:
        Configured optimizer instance with optional loaded state.
    """
    optimizer_cls_name = cfg.class_name
    lr = cfg.lr

    target_model = model
    if is_model_parallel(model):
        target_model = model.module

    if cfg.group is not None:
        parameters = get_param_group(target_model, cfg.group, lr)
    else:
        parameters = target_model.parameters()

    args = ConfigManager.to_object(cfg.args) if cfg.args is not None else {}

    optimizer = OPTIMIZER_REGISTRY.get(optimizer_cls_name)(parameters, lr=lr, **args)
    if cfg.checkpoint is not None:
        optimizer.load_state_dict(
            torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        )

    return optimizer
