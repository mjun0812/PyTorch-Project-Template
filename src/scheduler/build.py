import torch
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from ..config import ConfigManager, LrSchedulerConfig, LrSchedulersConfig
from ..utils import Registry

SCHEDULER_REGISTRY = Registry("SCHEDULER")
"""Registry for learning rate scheduler classes."""


def build_lr_scheduler(
    cfg: LrSchedulersConfig, optimizer: optim.Optimizer, epoch: int, max_iter: int
) -> tuple[_LRScheduler | None, _LRScheduler | None]:
    """Build learning rate schedulers for training.

    Creates both iteration-based and epoch-based schedulers based on configuration.

    Args:
        cfg: Scheduler configuration containing iter and epoch scheduler settings.
        optimizer: PyTorch optimizer to schedule.
        epoch: Total number of epochs for epoch-based scheduling.
        max_iter: Maximum number of iterations for iteration-based scheduling.

    Returns:
        Tuple of (iter_scheduler, epoch_scheduler), either can be None.
    """
    epoch_scheduler, iter_scheduler = None, None

    if cfg.epoch_scheduler:
        epoch_scheduler = get_lr_scheduler(optimizer, cfg.epoch_scheduler, epoch)

    if cfg.iter_scheduler:
        iter_scheduler = get_lr_scheduler(optimizer, cfg.iter_scheduler, max_iter)

    return iter_scheduler, epoch_scheduler


def get_lr_scheduler(
    optimizer: optim.Optimizer, cfg: LrSchedulerConfig, num_loop: int
) -> _LRScheduler:
    """Create a learning rate scheduler instance.

    Handles scheduler-specific parameter adjustments and automatic
    calculation of step sizes based on total iterations/epochs.

    Args:
        optimizer: PyTorch optimizer to schedule.
        cfg: Scheduler configuration with class name and arguments.
        num_loop: Total number of loops (epochs or iterations).

    Returns:
        Configured learning rate scheduler instance.
    """
    args = cfg.args
    if args and not isinstance(args, dict):
        args = ConfigManager.to_object(args)
    elif args is None:
        args = {}

    if cfg.class_name == "CosineLRScheduler":
        if args.get("noise_range_t") is not None:
            noise_range_t = []
            for n in args["noise_range_t"]:
                if isinstance(n, float):
                    noise_range_t.append(round(n * num_loop))
                elif isinstance(n, int):
                    noise_range_t.append(n)
            args["noise_range_t"] = noise_range_t
        args["t_initial"] = num_loop
    elif cfg.class_name == "PolynomialLRDecay":
        args["max_decay_steps"] = num_loop
    elif cfg.class_name == "MultiStepLR":
        milestones = []
        for m in args["milestones"]:
            if isinstance(m, float):
                milestones.append(round(m * num_loop))
            elif isinstance(m, int):
                milestones.append(m)
        args["milestones"] = milestones
    elif cfg.class_name == "StepLR":
        if isinstance(args["step_size"], float) and args["step_size"] <= 1:
            args["step_size"] = round(args["step_size"] * num_loop)
    elif cfg.class_name == "LinearLR":
        if args.get("total_iters") is None:
            args["total_iters"] = num_loop
    elif cfg.class_name == "ChainedScheduler":
        schedulers = []
        for s in args["schedulers"]:
            s = LrSchedulerConfig(**s)
            schedulers.append(get_lr_scheduler(optimizer, s, num_loop))
        args["schedulers"] = schedulers

    scheduler = SCHEDULER_REGISTRY.get(cfg.class_name)(optimizer=optimizer, **args)

    if cfg.checkpoint is not None:
        scheduler.load_state_dict(
            torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        )

    return scheduler
