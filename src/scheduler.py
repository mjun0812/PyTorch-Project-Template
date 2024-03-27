import math

import torch.optim as optim
from timm.scheduler import CosineLRScheduler


def build_lr_scheduler(cfg, optimizer, epoch, num_one_epoch_iters, logger=None):
    iter_scheduler = None
    lr_scheduler_name = cfg.NAME
    if lr_scheduler_name == "ReduceLROnPlateau":
        # factor : 学習率の減衰率
        # patience : 何ステップ向上しなければlrを変更するか
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=cfg.PATIENCE,
            verbose=True,
            threshold=cfg.THRESHOLD,
            mode="min",
            factor=cfg.FACTOR,
            min_lr=cfg.MIN_LR,
            cooldown=cfg.COOLDOWN,
        )
    elif lr_scheduler_name == "CosineAnnealingWarmRestarts":
        # T_0を周期とするコサインカーブで減衰して、
        # あるところまで減衰したところで再び高いlearning rateに戻すような挙動により
        # 局所最適を脱出してもっと良いパラメータを探索します
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.T_ZERO,
            T_mult=cfg.T_MULT,
            eta_min=cfg.ETA_MIN,
            verbose=False,
        )
    elif lr_scheduler_name == "CosineAnnealingWarmupReduceRestarts":
        scheduler = CosineAnnealingWarmupReduceRestarts(
            optimizer,
            first_cycle_steps=cfg.FIRST_CYCLE_STEPS,
            cycle_mult=cfg.CYCLE_MULT,
            max_lr=cfg.MAX_LR,
            min_lr=cfg.MIN_LR,
            gamma=cfg.GAMMA,
            warmup_steps=cfg.WARMUP_STEPS,
        )
    elif lr_scheduler_name == "CosineLRScheduler":
        if cfg.get("NOISE_T"):
            kwargs = {"noise_range_t": [int(epoch * n) for n in cfg.NOISE_T]}
        else:
            kwargs = {}
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epoch,
            lr_min=cfg.LR_MIN,
            warmup_t=cfg.WARMUP_T,
            warmup_lr_init=cfg.WARMUP_LR_INIT,
            cycle_limit=1,
            warmup_prefix=cfg.WARMUP_PREFIX,
            cycle_decay=cfg.CYCLE_DECAY,
            **kwargs,
        )
    elif lr_scheduler_name == "PolynomialLRDecay":
        scheduler = PolynomialLRDecay(
            optimizer,
            max_decay_steps=epoch,
            end_learning_rate=cfg.END_LR,
            power=cfg.POWER,
        )
    elif lr_scheduler_name == "MultiStepLR":
        milestones = []
        for m in cfg.MILESTONES:
            if isinstance(m, float):
                milestones.append(round(m * epoch))
            elif isinstance(m, int):
                milestones.append(m)
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=cfg.GAMMA,
        )
    elif lr_scheduler_name == "StepLR":
        if isinstance(cfg.LR_DROP, float):
            step_size = round(cfg.LR_DROP * epoch)
        else:
            step_size = cfg.LR_DROP
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=cfg.GAMMA,
        )
    elif lr_scheduler_name == "LinearLR":
        scheduler = LinearLR(
            optimizer,
            start_factor=cfg.START_FACTOR,
            end_factor=cfg.END_FACTOR,
            total_iters=cfg.TOTAL_ITERS,
        )
    elif lr_scheduler_name == "mmdet_1x":
        iter_scheduler = LinearLR(
            optimizer,
            start_factor=cfg.ITER_SCHEDULER.START_FACTOR,
            total_iters=cfg.ITER_SCHEDULER.TOTAL_ITERS,
        )
        milestones = []
        for m in cfg.EPOCH_SCHEDULER.MILESTONES:
            if isinstance(m, float):
                milestones.append(round(m * epoch))
            elif isinstance(m, int):
                milestones.append(m)
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=cfg.EPOCH_SCHEDULER.GAMMA,
        )
    elif lr_scheduler_name == "ChainedScheduler":
        schedulers = [
            build_lr_scheduler(c, optimizer, epoch, num_one_epoch_iters)[1] for c in cfg.SCHEDULERS
        ]
        scheduler = ChainedScheduler(schedulers)
    elif lr_scheduler_name == "IterScheduler":
        if cfg.ITER_SCHEDULER:
            _, iter_scheduler = build_lr_scheduler(
                cfg.ITER_SCHEDULER, optimizer, epoch * num_one_epoch_iters, num_one_epoch_iters
            )
        else:
            iter_scheduler = None
        if cfg.EPOCH_SCHEDULER:
            _, scheduler = build_lr_scheduler(
                cfg.EPOCH_SCHEDULER, optimizer, epoch, num_one_epoch_iters
            )
        else:
            scheduler = None

    if logger is not None:
        logger.info(f"LR Scheduler: {cfg.NAME}")
    return iter_scheduler, scheduler


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
        end_learning_rate: scheduler stoping learning rate decay,
        value of learning rate must be this value
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
        super().step(metric, epoch=None)


class CosineAnnealingWarmRestarts(optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def step(self, epoch=None, metric=None):
        super().step(epoch=epoch)


class MultiStepLR(optim.lr_scheduler.MultiStepLR):
    def step(self, epoch=None, metric=None):
        super().step()


class StepLR(optim.lr_scheduler.StepLR):
    def step(self, epoch=None, metric=None):
        super().step()


class LinearLR(optim.lr_scheduler.LinearLR):
    def step(self, epoch=None, metric=None):
        super().step()


class ChainedScheduler(optim.lr_scheduler.ChainedScheduler):
    def step(self, epoch=None, metric=None):
        super().step()
