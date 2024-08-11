from typing import Optional

from loguru import logger
from timm.scheduler import CosineLRScheduler
from torch import optim

from ..config import ExperimentConfig, LrSchedulerConfig
from .scheduler import (
    ChainedScheduler,
    CosineAnnealingWarmRestarts,
    CosineAnnealingWarmupReduceRestarts,
    LinearLR,
    MultiStepLR,
    PolynomialLRDecay,
    ReduceLROnPlateau,
    StepLR,
)


def build_lr_scheduler(
    cfg: ExperimentConfig, optimizer: optim.Optimizer
) -> tuple[Optional[optim.lr_scheduler._LRScheduler], Optional[optim.lr_scheduler._LRScheduler]]:
    iter_scheduler = None

    lr_scheduler_name = cfg.lr_scheduler.scheduler

    if lr_scheduler_name == "IterScheduler":
        if cfg.args.iter_scheduler:
            c = LrSchedulerConfig(**cfg.lr_scheduler.args["iter_scheduler"])
            iter_scheduler = get_lr_scheduler(optimizer, c, cfg.epoch)
        else:
            iter_scheduler = None

        if cfg.args.epoch_scheduler:
            c = LrSchedulerConfig(**cfg.lr_scheduler.args["epoch_scheduler"])
            scheduler = get_lr_scheduler(optimizer, c, cfg.epoch)
        else:
            scheduler = None
    else:
        scheduler = get_lr_scheduler(optimizer, cfg.lr_scheduler, cfg.epoch)

    logger.info(f"LR Scheduler: {lr_scheduler_name}")
    return iter_scheduler, scheduler


def get_lr_scheduler(
    optimizer: optim.Optimizer, cfg: LrSchedulerConfig, epoch: int
) -> optim.lr_scheduler._LRScheduler:
    lr_scheduler_name = cfg.scheduler
    args = cfg.get("args", {})

    if lr_scheduler_name == "ReduceLROnPlateau":
        # factor : 学習率の減衰率
        # patience : 何ステップ向上しなければlrを変更するか
        scheduler = ReduceLROnPlateau(optimizer, **args)
    elif lr_scheduler_name == "CosineAnnealingWarmRestarts":
        # T_0を周期とするコサインカーブで減衰して、
        # あるところまで減衰したところで再び高いlearning rateに戻すような挙動により
        # 局所最適を脱出してもっと良いパラメータを探索します
        scheduler = CosineAnnealingWarmRestarts(optimizer, **args)
    elif lr_scheduler_name == "CosineAnnealingWarmupReduceRestarts":
        scheduler = CosineAnnealingWarmupReduceRestarts(optimizer, **args)
    elif lr_scheduler_name == "CosineLRScheduler":
        if args.get("noise_range_t"):
            args["noise_range_t"] = [int(epoch * n) for n in cfg.NOISE_T]
        scheduler = CosineLRScheduler(optimizer, t_initial=epoch, **args)
    elif lr_scheduler_name == "PolynomialLRDecay":
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=epoch, **args)
    elif lr_scheduler_name == "MultiStepLR":
        milestones = []
        for m in args["milestones"]:
            if isinstance(m, float):
                milestones.append(round(m * epoch))
            elif isinstance(m, int):
                milestones.append(m)
        args["milestones"] = milestones
        scheduler = MultiStepLR(optimizer, **args)
    elif lr_scheduler_name == "StepLR":
        if isinstance(args["lr_drop"], float):
            args["lr_drop"] = round(args["lr_drop"] * epoch)
        else:
            args["lr_drop"] = args["lr_drop"]
        scheduler = StepLR(optimizer, **args)
    elif lr_scheduler_name == "LinearLR":
        scheduler = LinearLR(optimizer, **args)
    elif lr_scheduler_name == "ChainedScheduler":
        schedulers = []
        for s in cfg.schedulers:
            s = LrSchedulerConfig(**s)
            schedulers.append(get_lr_scheduler(optimizer, s, epoch))
        scheduler = ChainedScheduler(schedulers)

    return scheduler
