from copy import deepcopy
from typing import Optional

import torch
from loguru import logger
from timm.scheduler import create_scheduler_v2
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from ..config import ConfigManager, ExperimentConfig, LrSchedulerConfig
from ..utils import Registry

SCHEDULER_REGISTRY = Registry("SCHEDULER")


def build_lr_scheduler(
    cfg: ExperimentConfig, optimizer: optim.Optimizer
) -> tuple[Optional[_LRScheduler], Optional[_LRScheduler]]:
    lr_schedulers_cfg = cfg.lr_scheduler
    epoch_scheduler, iter_scheduler = None, None

    if lr_schedulers_cfg.epoch_scheduler:
        epoch_scheduler = get_lr_scheduler(optimizer, lr_schedulers_cfg.epoch_scheduler, cfg.epoch)
        logger.info(f"Epoch Scheduler: {lr_schedulers_cfg.epoch_scheduler.scheduler}")

    if lr_schedulers_cfg.iter_scheduler:
        iter_scheduler = get_lr_scheduler(optimizer, lr_schedulers_cfg.iter_scheduler, cfg.max_iter)
        logger.info(f"Iter Scheduler: {lr_schedulers_cfg.iter_scheduler.scheduler}")

    return iter_scheduler, epoch_scheduler


def get_lr_scheduler(
    optimizer: optim.Optimizer, cfg: LrSchedulerConfig, num_loop: int
) -> _LRScheduler:
    lr_scheduler_name = cfg.scheduler

    args = cfg.get("args")
    if args and not isinstance(args, dict):
        args = ConfigManager.to_object(deepcopy(args))
    elif args is None:
        args = {}

    if lr_scheduler_name == "CosineLRScheduler":
        if args.get("noise_range_t") is not None:
            noise_range_t = []
            for n in args["noise_range_t"]:
                if isinstance(n, float):
                    noise_range_t.append(round(n * num_loop))
                elif isinstance(n, int):
                    noise_range_t.append(n)
            args["noise_range_t"] = noise_range_t
        args["t_initial"] = num_loop
    elif lr_scheduler_name == "PolynomialLRDecay":
        args["max_decay_steps"] = num_loop
    elif lr_scheduler_name == "MultiStepLR":
        milestones = []
        for m in args["milestones"]:
            if isinstance(m, float):
                milestones.append(round(m * num_loop))
            elif isinstance(m, int):
                milestones.append(m)
        args["milestones"] = milestones
    elif lr_scheduler_name == "StepLR":
        if isinstance(args["step_size"], float) and args["step_size"] <= 1:
            args["step_size"] = round(args["step_size"] * num_loop)
    elif lr_scheduler_name == "LinearLR":
        if args.get("total_iters") is None:
            args["total_iters"] = num_loop
    elif lr_scheduler_name == "ChainedScheduler":
        schedulers = []
        for s in args["schedulers"]:
            s = LrSchedulerConfig(**s)
            schedulers.append(get_lr_scheduler(optimizer, s, num_loop))
        args["schedulers"] = schedulers

    if "timm" in lr_scheduler_name:
        scheduler, _ = create_scheduler_v2(
            optimizer=optimizer, sched=lr_scheduler_name.replace("_timm", ""), **args
        )
    else:
        scheduler = SCHEDULER_REGISTRY.get(lr_scheduler_name)(optimizer=optimizer, **args)

    if cfg.checkpoint is not None:
        scheduler.load_state_dict(torch.load(cfg.checkpoint, map_location="cpu", weights_only=True))

    return scheduler
