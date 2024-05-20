from typing import Literal

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import TORCH_DTYPE, reduce_tensor


class Trainer:
    def __init__(
        self,
        rank: int,
        epochs: int,
        device: torch.device,
        use_amp: bool,
        optimizer,
        lr_scheduler,
        use_clip_grad=True,
        clip_grad=10.0,
        iter_lr_scheduler=None,
        amp_init_scale=False,
        amp_dtype="fp16",
    ) -> None:
        super().__init__()
        self.rank = rank
        self.device = device
        self.is_cpu = device.type == "cpu"
        self.use_amp = use_amp
        self.amp_dtype = TORCH_DTYPE[amp_dtype] if use_amp else torch.float32
        if amp_init_scale:
            self.scaler = torch.cuda.amp.GradScaler(init_scale=amp_init_scale, enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.optimizer = optimizer
        self.iter_lr_scheduler = iter_lr_scheduler
        self.lr_scheduler = lr_scheduler
        self.use_clip_grad = use_clip_grad
        self.clip_grad = clip_grad

        self.epochs = epochs

    def do_one_epoch(
        self,
        phase: Literal["train", "val"],
        epoch: int,
        model: torch.nn.Module,
        criterion,
        dataloader: DataLoader,
        batched_transform,
        evaluator=None,
        model_ema=None,
    ):
        hist_epoch_loss = {}

        # Set progress bar
        progress_bar = enumerate(dataloader)
        if self.rank in [-1, 0]:
            progress_bar = tqdm(progress_bar, total=len(dataloader), dynamic_ncols=True)

        model.train(phase == "train")
        if self.rank != -1:
            model.module.phase = phase
        else:
            model.phase = phase

        for i, data in progress_bar:
            with torch.set_grad_enabled(phase == "train"):
                # ####### Prepare Input Data #######
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device, non_blocking=True)
                if batched_transform:
                    data = batched_transform(data)

                # ####### Forward #######
                with torch.autocast(
                    device_type="cuda" if not self.is_cpu else "cpu",
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):
                    output = model(data)
                    loss = criterion(output, data)

                # ####### Backward #######
                if phase == "train":
                    self.scaler.scale(loss["total_loss"]).backward()
                    if self.use_clip_grad:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.iter_lr_scheduler:
                        self.iter_lr_scheduler.step(
                            epoch=i + epoch * len(dataloader), metric=loss["total_loss"].item()
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    if model_ema:
                        model_ema.update(model)

                if evaluator:
                    evaluator.update(*self.generate_input_evaluator(output, data))
                for key in loss.keys():
                    if self.rank != -1:
                        loss[key] = reduce_tensor(loss[key]) / dist.get_world_size()
                    hist_epoch_loss[key] = (
                        hist_epoch_loss.get(key, 0.0) + loss[key].item() * dataloader.batch_size
                    )
                if self.rank in [-1, 0]:
                    lr = self.optimizer.param_groups[0]["lr"]
                    description = f"Epoch: {epoch + 1:3}/{self.epochs:3}. LR: {lr:.4e}"
                    for k, v in loss.items():
                        description += f" {k.capitalize()}: {v.item():8.4f}"
                    progress_bar.set_description(description)

        # Finish Epoch Process below
        for k, v in hist_epoch_loss.items():
            hist_epoch_loss[k] = hist_epoch_loss[k] / (len(dataloader) * dataloader.batch_size)
        lr = self.optimizer.param_groups[0]["lr"]
        if phase == "train" and self.lr_scheduler:
            self.lr_scheduler.step(epoch=epoch, metric=hist_epoch_loss["total_loss"])

        ret = {"epoch_losses": hist_epoch_loss, "lr": lr}
        if evaluator:
            metrics = evaluator.compute()
            evaluator.reset()
            ret["metrics"] = metrics
        return ret

    def build_epoch_log(
        self,
        phase: Literal["train", "val"],
        epoch: int,
        epoch_losses: dict,
        lr: float,
        **kwargs,
    ):
        description = (
            f"{phase.capitalize()} Epoch: {(epoch + 1):3}/{self.epochs:3}. "
            f"GPU: {torch.cuda.memory_reserved(self.device) / 1e9:.1f}GB. "
            f"LR: {lr:.4e}"
        )
        for k, v in epoch_losses.items():
            description += f" {k}: {v:8.4f}"
        return description

    def generate_input_evaluator(self, output, data):
        return output, data
