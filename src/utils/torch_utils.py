import logging
import math

import torch.optim as optim


logger = logging.getLogger()


def build_optimizer(cfg, model):
    optimizer_name = cfg.OPTIMIZER
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=cfg.LR, momentum=0.937, nesterov=True)
    logger.info(f"Using Optimizer is {optimizer_name}")
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    lr_scheduler_name = cfg.LR_SCHEDULER
    if lr_scheduler_name == "reduce":
        # factor : 学習率の減衰率
        # patience : 何ステップ向上しなければlrを変更するか
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=cfg.EPOCH // 10,
            verbose=True,
            mode="min",
            factor=0.5,
            min_lr=1e-6,
            cooldown=cfg.EPOCH // 10,
        )
    elif lr_scheduler_name == "sgdr":
        # T_0を周期とするコサインカーブで減衰して、
        # あるところまで減衰したところで再び高いlearning rateに戻すような挙動により
        # 局所最適を脱出してもっと良いパラメータを探索します
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6, verbose=False
        )
    elif lr_scheduler_name == "sgdr_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, first_cycle_steps=10, cycle_mult=2, max_lr=0.1, min_lr=1e-6, gamma=0.5, warmup_steps=5
        )
    logger.info(f"Using LR Scheduler is {cfg.LR_SCHEDULER}")
    return scheduler


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
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

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

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

    def step(self, epoch=None):
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
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
