from dataclasses import dataclass, field
from typing import Optional, TypedDict

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torchmetrics import MetricCollection
from tqdm import tqdm

from .models import BaseModel
from .types import TORCH_DTYPE, ModelOutput, PhaseStr
from .utils import get_world_size, is_distributed, is_local_main_process, reduce_tensor

try:
    from torch import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class HistoryEpochLoss(TypedDict, total=False):
    total_loss: float


@dataclass
class EpochResult:
    epoch_losses: HistoryEpochLoss = field(default_factory=HistoryEpochLoss)
    lr: float = 0.0
    metrics: Optional[dict] = None


class BaseTrainer:
    def __init__(
        self,
        epochs: int,
        device: torch.device,
        datasets: dict[PhaseStr, torch.utils.data.Dataset],
        dataloaders: dict[PhaseStr, torch.utils.data.DataLoader],
        batched_transforms: dict[PhaseStr, callable],
        optimizer: torch.optim.Optimizer,
        epoch_lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        iter_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        evaluators: Optional[dict[PhaseStr, MetricCollection]] = None,
        generate_input_evaluator: Optional[callable] = None,
        use_clip_grad: bool = True,
        clip_grad: float = 10.0,
        gpu_multi_strategy: str = "ddp",
        use_amp: bool = False,
        amp_init_scale: int = 2**16,
        amp_dtype: str = "fp16",
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.epochs = epochs
        self.device = device
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.batched_transforms = batched_transforms
        self.optimizer = optimizer
        self.epoch_lr_scheduler = epoch_lr_scheduler
        self.iter_lr_scheduler = iter_lr_scheduler
        self.evaluators = evaluators
        self.generate_input_evaluator = (
            generate_input_evaluator or self.default_generate_input_evaluator
        )
        self.use_clip_grad = use_clip_grad
        self.clip_grad = clip_grad
        self.gpu_multi_strategy = gpu_multi_strategy
        self.use_amp = use_amp
        self.amp_dtype = TORCH_DTYPE[amp_dtype] if use_amp else torch.float32

        if self.use_amp:
            logger.info(f"Using Mixed Precision with AMP (dtype: {self.amp_dtype})")

        self.is_cpu = device.type == "cpu"

        if self.gpu_multi_strategy == "fsdp":
            self.scaler = ShardedGradScaler(init_scale=amp_init_scale, enabled=self.use_amp)
        else:
            self.scaler = GradScaler(init_scale=amp_init_scale, enabled=self.use_amp)

        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.gradient_accumulation_steps > 1:
            logger.info(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")

        for phase in ["train", "val"]:
            total_iters = len(self.dataloaders[phase]) * self.epochs
            logger.info(f"Total {phase} Iterations per GPU: {total_iters}")
            logger.info(f"Total {phase} Iterations per World: {total_iters * get_world_size()}")

    def do_one_epoch(
        self,
        phase: PhaseStr,
        current_epoch: int,
        model: BaseModel,
    ) -> EpochResult:
        hist_epoch_loss = HistoryEpochLoss()
        pbar = self._setup_progress_bar(phase)
        self._set_model_phase(model, phase, current_epoch)
        self.optimizer.zero_grad(set_to_none=True)

        # For ScheduleFree Optimizer
        if phase == "train" and getattr(self.optimizer, "train", False):
            self.optimizer.train()
        elif phase == "val" and getattr(self.optimizer, "eval", False):
            self.optimizer.eval()

        for i, data in pbar:
            with torch.set_grad_enabled(phase == "train"):
                data = self.prepare_input(data, phase)
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                    cache_enabled=False,
                ):
                    output: ModelOutput = model(data)
                if phase == "train":
                    self.backward(output, model, i, current_epoch)

            self.update_metrics_and_losses(phase, data, output, hist_epoch_loss)
            if is_local_main_process():
                self._update_pbar(pbar, current_epoch, output["losses"])

        return self.after_epoch(phase, current_epoch, hist_epoch_loss)

    def _setup_progress_bar(self, phase: PhaseStr):
        progress_bar = enumerate(self.dataloaders[phase])
        if is_local_main_process():
            progress_bar = tqdm(
                progress_bar, total=len(self.dataloaders[phase]), dynamic_ncols=True
            )
        return progress_bar

    def _set_model_phase(self, model: BaseModel, phase: PhaseStr, epoch: int):
        model.train(phase == "train")
        if is_distributed():
            model.module.phase = phase
            self.dataloaders[phase].sampler.set_epoch(epoch)
        else:
            model.phase = phase

    def prepare_input(self, data: dict, phase: PhaseStr) -> dict:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device, non_blocking=True)
        if self.batched_transforms[phase]:
            data = self.batched_transforms[phase](data)
        return data

    def backward(self, output: ModelOutput, model: BaseModel, iter_idx: int, epoch: int):
        total_loss = output["losses"]["total_loss"] / self.gradient_accumulation_steps
        self.scaler.scale(total_loss).backward()
        if (iter_idx + 1) % self.gradient_accumulation_steps == 0:
            if self.use_clip_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.iter_lr_scheduler:
                self.iter_lr_scheduler.step(
                    epoch=iter_idx + epoch * len(self.dataloaders["train"]),
                    metric=output["losses"]["total_loss"].item(),
                )
            self.optimizer.zero_grad(set_to_none=True)

    def update_metrics_and_losses(
        self, phase: PhaseStr, data: dict, output: ModelOutput, hist_epoch_loss: HistoryEpochLoss
    ):
        if self.evaluators[phase] is not None:
            evaluator_input = self.generate_input_evaluator(data, output["preds"])
            self.evaluators[phase].update(*evaluator_input)
        for key, value in output["losses"].items():
            if is_distributed():
                value = reduce_tensor(value) / dist.get_world_size()
            hist_epoch_loss[key] = hist_epoch_loss.get(key, 0.0) + value.item()

    def _update_pbar(self, pbar, epoch: int, losses: dict):
        lr = self.optimizer.param_groups[0]["lr"]
        description = f"Epoch: {epoch + 1:3}/{self.epochs:3}. LR: {lr:.4e}"
        for k, v in losses.items():
            description += f" {k.capitalize()}: {v.item():8.4f}"
        pbar.set_description(description)

    def after_epoch(
        self, phase: PhaseStr, epoch: int, hist_epoch_loss: HistoryEpochLoss
    ) -> EpochResult:
        for k, v in hist_epoch_loss.items():
            hist_epoch_loss[k] = v / len(self.dataloaders[phase])

        lr = self.optimizer.param_groups[0]["lr"]
        if phase == "train" and self.epoch_lr_scheduler:
            self.epoch_lr_scheduler.step(epoch=epoch, metric=hist_epoch_loss["total_loss"])

        ret = EpochResult(epoch_losses=hist_epoch_loss, lr=lr)
        if self.evaluators[phase] is not None:
            ret.metrics = self.evaluators[phase].compute()
            self.evaluators[phase].reset()
        logger.info(
            self._build_epoch_log(phase=phase, epoch=epoch, epoch_losses=hist_epoch_loss, lr=lr)
        )
        return ret

    def _build_epoch_log(
        self, phase: PhaseStr, epoch: int, epoch_losses: HistoryEpochLoss, lr: float
    ) -> str:
        description = [
            f"{phase.capitalize()} Epoch: {(epoch + 1):3}/{self.epochs:3}.",
            f"LR: {lr:.4e}",
        ]
        if self.device.type == "cuda":
            description.append(f"GPU: {torch.cuda.memory_reserved(self.device) / 1e9:.1f}GB.")
        for k, v in epoch_losses.items():
            description.append(f"{k}: {v:8.4f}")
        return " ".join(description)

    def default_generate_input_evaluator(self, targets, preds):
        return targets, preds


class Trainer(BaseTrainer):
    def default_generate_input_evaluator(self, targets, preds):
        return targets, preds
