from dataclasses import dataclass, field
from typing import Optional, TypedDict

import torch
import torch.distributed as dist
from loguru import logger
from torchmetrics import MetricCollection
from tqdm import tqdm

from .alias import PhaseStr
from .models import BaseModel, ModelOutput
from .utils import TORCH_DTYPE, is_distributed, is_main_process, reduce_tensor


class HistoryEpochLoss(TypedDict, total=False):
    total_loss: float = 0.0


@dataclass
class EpochResult:
    epoch_losses: HistoryEpochLoss = field(default_factory=HistoryEpochLoss)
    lr: float = 0.0
    metrics: Optional[dict] = None


class Trainer:
    def __init__(
        self,
        epochs: int,
        device: torch.device,
        use_amp: bool,
        datasets: dict[PhaseStr, torch.utils.data.Dataset],
        dataloaders: dict[PhaseStr, torch.utils.data.DataLoader],
        batched_transforms: dict[PhaseStr, callable],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        iter_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        evaluators: Optional[dict[PhaseStr, MetricCollection]] = None,
        use_clip_grad: bool = True,
        clip_grad: float = 10.0,
        amp_init_scale=None,
        amp_dtype: str = "fp16",
    ) -> None:
        self.device = device
        self.is_cpu = device.type == "cpu"
        self.amp_dtype = TORCH_DTYPE[amp_dtype] if use_amp else torch.float32
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.batched_transforms = batched_transforms
        self.optimizer = optimizer
        self.iter_lr_scheduler = iter_lr_scheduler
        self.lr_scheduler = lr_scheduler
        self.evaluators = evaluators
        self.use_clip_grad = use_clip_grad
        self.clip_grad = clip_grad
        self.epochs = epochs

        self.use_amp = use_amp
        if self.use_amp:
            logger.info("Using Mixed Precision with AMP")
            logger.info(f"AMP dtype: {self.amp_dtype}")
        if amp_init_scale:
            self.scaler = torch.cuda.amp.GradScaler(init_scale=amp_init_scale, enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def do_one_epoch(
        self,
        phase: PhaseStr,
        epoch: int,
        model: BaseModel,
        model_ema=None,
    ) -> EpochResult:
        hist_epoch_loss = HistoryEpochLoss()

        # Set progress bar
        progress_bar = enumerate(self.dataloaders[phase])
        if is_main_process():
            progress_bar = tqdm(
                progress_bar, total=len(self.dataloaders[phase]), dynamic_ncols=True
            )

        model.train(phase == "train")
        if is_distributed():
            model.module.phase = phase
            self.dataloaders[phase].sampler.set_epoch(epoch)
        else:
            model.phase = phase

        for i, data in progress_bar:
            with torch.set_grad_enabled(phase == "train"):
                # ####### Prepare Input Data #######
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device, non_blocking=True)
                if self.batched_transforms[phase]:
                    data = self.batched_transforms[phase](data)

                # ####### Forward #######
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):
                    output: ModelOutput = model(data)

                # ####### Backward #######
                if phase == "train":
                    self.scaler.scale(output["losses"]["total_loss"]).backward()
                    if self.use_clip_grad:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.iter_lr_scheduler:
                        self.iter_lr_scheduler.step(
                            epoch=i + epoch * len(self.dataloaders[phase]),
                            metric=output["losses"]["total_loss"].item(),
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    if model_ema:
                        model_ema.update(model)

            # ####### After each iteration #######
            if self.evaluators[phase]:
                self.evaluators[phase].update(*self.generate_input_evaluator(data, output["preds"]))
            for key in output["losses"].keys():
                if is_distributed():
                    output["losses"][key] = (
                        reduce_tensor(output["losses"][key]) / dist.get_world_size()
                    )
                hist_epoch_loss[key] = (
                    hist_epoch_loss.get(key, 0.0)
                    + output["losses"][key].item() * self.dataloaders[phase].batch_size
                )
            if is_main_process():
                lr = self.optimizer.param_groups[0]["lr"]
                description = f"Epoch: {epoch + 1:3}/{self.epochs:3}. LR: {lr:.4e}"
                for k, v in output["losses"].items():
                    description += f" {k.capitalize()}: {v.item():8.4f}"
                progress_bar.set_description(description)

        # ####### After each epoch #######
        for k, v in hist_epoch_loss.items():
            hist_epoch_loss[k] = hist_epoch_loss[k] / len(self.datasets[phase])

        # Epoch based lr_scheduler
        lr = self.optimizer.param_groups[0]["lr"]
        if phase == "train" and self.lr_scheduler:
            self.lr_scheduler.step(epoch=epoch, metric=hist_epoch_loss["total_loss"])

        ret = EpochResult(epoch_losses=hist_epoch_loss, lr=lr)
        if self.evaluators[phase]:
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

    def generate_input_evaluator(self, targets, preds):
        return targets, preds
