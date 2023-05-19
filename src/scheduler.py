import logging
import math

from timm.scheduler import CosineLRScheduler

import torch.optim as optim


logger = logging.getLogger()


def build_lr_scheduler(cfg, optimizer):
    lr_scheduler_name = cfg.LR_SCHEDULER.NAME
    if lr_scheduler_name == "ReduceLROnPlateau":
        # factor : 学習率の減衰率
        # patience : 何ステップ向上しなければlrを変更するか
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=cfg.LR_SCHEDULER.PATIENCE,
            verbose=True,
            threshold=cfg.LR_SCHEDULER.THRESHOLD,
            mode="min",
            factor=cfg.LR_SCHEDULER.FACTOR,
            min_lr=cfg.LR_SCHEDULER.MIN_LR,
            cooldown=cfg.LR_SCHEDULER.COOLDOWN,
        )
    elif lr_scheduler_name == "CosineAnnealingWarmRestarts":
        # T_0を周期とするコサインカーブで減衰して、
        # あるところまで減衰したところで再び高いlearning rateに戻すような挙動により
        # 局所最適を脱出してもっと良いパラメータを探索します
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.LR_SCHEDULER.T_ZERO,
            T_mult=cfg.LR_SCHEDULER.T_MULT,
            eta_min=cfg.LR_SCHEDULER.ETA_MIN,
            verbose=False,
        )
    elif lr_scheduler_name == "CosineAnnealingWarmupReduceRestarts":
        scheduler = CosineAnnealingWarmupReduceRestarts(
            optimizer,
            first_cycle_steps=cfg.LR_SCHEDULER.FIRST_CYCLE_STEPS,
            cycle_mult=cfg.LR_SCHEDULER.CYCLE_MULT,
            max_lr=cfg.LR_SCHEDULER.MAX_LR,
            min_lr=cfg.LR_SCHEDULER.MIN_LR,
            gamma=cfg.LR_SCHEDULER.GAMMA,
            warmup_steps=cfg.LR_SCHEDULER.WARMUP_STEPS,
        )
    elif lr_scheduler_name == "CosineLRScheduler":
        if cfg.LR_SCHEDULER.get("NOISE_T"):
            kwargs = {"noise_range_t": [int(cfg.EPOCH * n) for n in cfg.LR_SCHEDULER.NOISE_T]}
        else:
            kwargs = {}
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.LR_SCHEDULER.T_INITIAL,
            lr_min=cfg.LR_SCHEDULER.LR_MIN,
            warmup_t=cfg.LR_SCHEDULER.WARMUP_T,
            warmup_lr_init=cfg.LR_SCHEDULER.WARMUP_LR_INIT,
            cycle_limit=1,
            warmup_prefix=cfg.LR_SCHEDULER.WARMUP_PREFIX,
            cycle_decay=cfg.LR_SCHEDULER.CYCLE_DECAY,
            **kwargs,
        )
    elif lr_scheduler_name == "PolynomialLRDecay":
        scheduler = PolynomialLRDecay(
            optimizer,
            max_decay_steps=cfg.LR_SCHEDULER.MAX_DECAY_STEPS,
            end_learning_rate=cfg.LR_SCHEDULER.END_LR,
            power=cfg.LR_SCHEDULER.POWER,
        )
    elif lr_scheduler_name == "MultiStepLR":
        scheduler = MultiStepLR(
            optimizer,
            milestones=[round(r * cfg.EPOCH) for r in cfg.LR_SCHEDULER.MILESTONES],
            gamma=cfg.LR_SCHEDULER.GAMMA,
        )
    elif lr_scheduler_name == "StepLR":
        scheduler = StepLR(
            optimizer,
            step_size=round(cfg.LR_SCHEDULER.LR_DROP * cfg.EPOCH),
            gamma=cfg.LR_SCHEDULER.GAMMA,
        )
    elif lr_scheduler_name == "LinearLR":
        scheduler = LinearLR(
            optimizer,
            start_factor=cfg.LR_SCHEDULER.START_FACTOR,
            end_factor=cfg.LR_SCHEDULER.END_FACTOR,
            total_iters=cfg.LR_SCHEDULER.TOTAL_ITERS,
        )

    logger.info(f"LR Scheduler: {cfg.LR_SCHEDULER}")
    return scheduler


class CosineAnnealingWarmupReduceRestarts(optim.lr_scheduler._LRScheduler):
    """https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup"""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 1e-4,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            first_cycle_steps (int): First cycle step size.
            cycle_mult(float): Cycle steps magnification. Default: -1.
            max_lr(float): First cycle's max learning rate. Default: 0.1.
            min_lr(float): Min learning rate. Default: 0.001.
            warmup_steps(int): Linear warmup step size. Default: 0.
            gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupReduceRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None, metric=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1

            if self.step_in_cycle >= self.cur_cycle_steps:
                # update cycle size
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult**n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class PolynomialLRDecay(optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.0:
            raise ValueError("max_decay_steps should be greater than 1.")
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [
            (base_lr - self.end_learning_rate)
            * ((1 - self.last_step / self.max_decay_steps) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metric=None):
        if epoch is None:
            epoch = self.last_step + 1
        self.last_step = epoch if epoch != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [
                (base_lr - self.end_learning_rate)
                * ((1 - self.last_step / self.max_decay_steps) ** (self.power))
                + self.end_learning_rate
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group["lr"] = lr


class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    def step(self, epoch=None, metric=None):
        super().step(self, metric, epoch=epoch)


class CosineAnnealingWarmRestarts(optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def step(self, epoch=None, metric=None):
        super().step(epoch=epoch)


class MultiStepLR(optim.lr_scheduler.MultiStepLR):
    def step(self, epoch=None, metric=None):
        super().step(epoch=epoch)


class StepLR(optim.lr_scheduler.StepLR):
    def step(self, epoch=None, metric=None):
        super().step(epoch=epoch)


class LinearLR(optim.lr_scheduler.LinearLR):
    def step(self, epoch=None, metric=None):
        super().step(epoch=epoch)
