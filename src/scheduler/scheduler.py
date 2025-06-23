import math
from collections.abc import Sequence
from typing import Any

from torch import optim

from .build import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register()
class CosineAnnealingWarmupReduceRestarts(optim.lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with warmup and restarts.

    Implements cosine annealing with linear warmup and cycling restarts.
    Based on: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

    Attributes:
        first_cycle_steps: Number of steps in the first cycle.
        cycle_mult: Multiplier for cycle length after each restart.
        base_max_lr: Base maximum learning rate.
        max_lr: Current maximum learning rate.
        min_lr: Minimum learning rate.
        warmup_steps: Number of warmup steps.
        gamma: Decay factor for maximum learning rate after each cycle.
        cur_cycle_steps: Current cycle step count.
        cycle: Current cycle number.
        step_in_cycle: Current step within the cycle.
    """

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
    ) -> None:
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

        super().__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self) -> list[float]:
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

    def step(self, epoch: int | None = None, metric: Any | None = None) -> None:
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
        elif epoch >= self.first_cycle_steps:
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


@SCHEDULER_REGISTRY.register()
class PolynomialLRDecay(optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate decay scheduler.

    Decays learning rate using polynomial function until reaching
    max_decay_steps, then maintains end_learning_rate.

    Attributes:
        max_decay_steps: Maximum number of decay steps.
        end_learning_rate: Final learning rate after decay.
        power: Power of the polynomial decay function.
        last_step: Current step number.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_decay_steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
    ) -> None:
        if max_decay_steps <= 1.0:
            raise ValueError("max_decay_steps should be greater than 1.")
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [
            (base_lr - self.end_learning_rate)
            * ((1 - self.last_step / self.max_decay_steps) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: int | None = None, metric: Any | None = None) -> None:
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


@SCHEDULER_REGISTRY.register()
class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    """Reduce learning rate on plateau with custom step interface.

    Extends PyTorch's ReduceLROnPlateau to match the common step interface
    used by other schedulers in the framework.
    """

    def step(self, epoch: int | None = None, metric: Any | None = None) -> None:
        """Step the scheduler.

        Args:
            epoch: Current epoch (unused for this scheduler).
            metric: Metric value to monitor for plateau detection.
        """
        super().step(metric, epoch=None)


@SCHEDULER_REGISTRY.register()
class CosineAnnealingWarmRestarts(optim.lr_scheduler.CosineAnnealingWarmRestarts):
    """Cosine annealing with warm restarts.

    Extends PyTorch's CosineAnnealingWarmRestarts to match the common
    step interface used by other schedulers in the framework.
    """

    def step(self, epoch: int | None = None, metric: Any | None = None) -> None:
        """Step the scheduler.

        Args:
            epoch: Current epoch for scheduling.
            metric: Metric value (unused for this scheduler).
        """
        super().step(epoch=epoch)


@SCHEDULER_REGISTRY.register()
class MultiStepLR(optim.lr_scheduler.MultiStepLR):
    """Multi-step learning rate scheduler.

    Extends PyTorch's MultiStepLR to match the common step interface
    used by other schedulers in the framework.
    """

    def step(self, epoch: int | None = None, metric: Any | None = None) -> None:
        """Step the scheduler.

        Args:
            epoch: Current epoch (unused, uses internal counter).
            metric: Metric value (unused for this scheduler).
        """
        super().step()


@SCHEDULER_REGISTRY.register()
class StepLR(optim.lr_scheduler.StepLR):
    """Step learning rate scheduler.

    Extends PyTorch's StepLR to match the common step interface
    used by other schedulers in the framework.
    """

    def step(self, epoch: int | None = None, metric: Any | None = None) -> None:
        """Step the scheduler.

        Args:
            epoch: Current epoch (unused, uses internal counter).
            metric: Metric value (unused for this scheduler).
        """
        super().step()


@SCHEDULER_REGISTRY.register()
class LinearLR(optim.lr_scheduler.LinearLR):
    """Linear learning rate scheduler.

    Extends PyTorch's LinearLR to match the common step interface
    used by other schedulers in the framework.
    """

    def step(self, epoch: int | None = None, metric: Any | None = None) -> None:
        """Step the scheduler.

        Args:
            epoch: Current epoch (unused, uses internal counter).
            metric: Metric value (unused for this scheduler).
        """
        super().step()


@SCHEDULER_REGISTRY.register()
class ChainedScheduler(optim.lr_scheduler.ChainedScheduler):
    """Chained scheduler with step range support.

    Extends PyTorch's ChainedScheduler to support step ranges,
    allowing different schedulers to be active during different
    training phases.

    Attributes:
        step_ranges: List of (start, end) tuples defining when each scheduler is active.
    """

    def __init__(
        self,
        schedulers: Sequence[optim.lr_scheduler._LRScheduler],
        optimizer: optim.Optimizer | None = None,
        step_ranges: Sequence[int] | None = None,
    ) -> None:
        """Initialize the chained scheduler.

        Args:
            schedulers: Sequence of schedulers to chain.
            optimizer: Optimizer to schedule (optional).
            step_ranges: Step ranges for each scheduler.
        """
        self.step_ranges = step_ranges
        super().__init__(schedulers, optimizer)

    def step(self, epoch: int | None = None, metric: Any | None = None) -> None:
        """Step the active schedulers based on current epoch.

        Args:
            epoch: Current epoch.
            metric: Metric value to pass to schedulers.
        """
        for scheduler, step_range in zip(self._schedulers, self.step_ranges):
            if step_range and step_range[0] <= epoch < step_range[1]:
                scheduler.step(epoch=epoch, metric=metric)
        self._last_lr = [group["lr"] for group in self._schedulers[-1].optimizer.param_groups]
