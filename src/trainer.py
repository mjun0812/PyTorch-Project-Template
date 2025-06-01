from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torchmetrics import MetricCollection
from tqdm import tqdm

from .config import ConfigManager, ExperimentConfig
from .models import BaseModel
from .types import TORCH_DTYPE, DatasetOutput, ModelOutput, PathLike, PhaseStr
from .utils import (
    Logger,
    create_symlink,
    is_distributed,
    is_local_main_process,
    is_world_main_process,
    load_model_weight,
    reduce_tensor,
    save_lr_scheduler,
    save_model,
    save_optimizer,
)

try:
    from torch import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class EpochResult(TypedDict, total=False):
    total_loss: float


@dataclass
class TrainingState:
    phase: PhaseStr
    current_epoch: int
    best_epoch: int
    best_metric: float
    progress_bar: tqdm | Iterator | None


@dataclass
class TrainerParams:
    epochs: int
    output_dir: Path
    use_clip_grad: bool = True
    clip_grad: float = 10.0
    gpu_multi_strategy: str = "ddp"
    use_amp: bool = False
    amp_init_scale: int = 2**16
    amp_dtype: str = "fp16"
    gradient_accumulation_steps: int = 1
    greater_is_better: bool = True
    metric_for_best_model: str = "total_loss"
    save_interval: int = 1
    val_interval: int = 1


class BaseTrainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        params: TrainerParams,
        device: torch.device,
        logger: Logger,
        model: BaseModel,
        datasets: dict[PhaseStr, torch.utils.data.Dataset],
        dataloaders: dict[PhaseStr, torch.utils.data.DataLoader],
        samplers: dict[PhaseStr, torch.utils.data.Sampler | None],
        batched_transforms: dict[PhaseStr, callable],
        optimizer: torch.optim.Optimizer,
        epoch_lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        iter_lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        evaluators: dict[PhaseStr, MetricCollection] | None = None,
    ) -> None:
        self.cfg = cfg
        self.params = params
        self.device = device
        self.logger = logger

        self.model = model
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.samplers = samplers
        self.batched_transforms = batched_transforms
        self.optimizer = optimizer
        self.epoch_lr_scheduler = epoch_lr_scheduler
        self.iter_lr_scheduler = iter_lr_scheduler
        self.evaluators = evaluators

        if self.params.gpu_multi_strategy == "fsdp":
            self.scaler = ShardedGradScaler(
                init_scale=self.params.amp_init_scale, enabled=self.params.use_amp
            )
        else:
            self.scaler = GradScaler(
                init_scale=self.params.amp_init_scale, enabled=self.params.use_amp
            )

        if self.params.greater_is_better:
            best_metric = 0
        else:
            best_metric = 1e8
        self.training_state = TrainingState(
            phase="train",
            current_epoch=0,
            best_metric=best_metric,
            best_epoch=0,
            progress_bar=None,
        )

    def _before_train(self, state: TrainingState):
        pass

    def train(self, start_epoch: int):
        self._before_train(self.training_state)

        for epoch in range(start_epoch, self.params.epochs):
            self.training_state.current_epoch = epoch + 1
            self.training_state.phase = "train"
            self.logger.phase = "train"
            self.logger.log_metric("Epoch", epoch + 1, epoch + 1)
            self._before_epoch(self.training_state)

            epoch_result = self._do_one_epoch(self.training_state)

            epoch_result = self._after_epoch(self.training_state, epoch_result)
            self.logger.log_metrics(epoch_result, epoch + 1, "train")
            self.logger.log_artifact(self.logger.log_path)

            if (epoch + 1) % self.params.save_interval == 0 or (epoch + 1) == self.params.epochs:
                self.save_state(epoch + 1, self.params.output_dir)

            # Validation
            if (epoch + 1) % self.params.val_interval == 0 or (epoch + 1) == self.params.epochs:
                self.training_state.phase = "val"
                self.logger.phase = "val"
                self._before_epoch(self.training_state)

                epoch_result = self._do_one_epoch(self.training_state)

                epoch_result = self._after_epoch(self.training_state, epoch_result)
                self.logger.log_metrics(epoch_result, epoch + 1, "val")
                self.logger.log_artifact(self.logger.log_path)

                if is_world_main_process():
                    # Update best model
                    metric_value = epoch_result[self.params.metric_for_best_model]
                    if (
                        self.params.greater_is_better
                        and metric_value > self.training_state.best_metric
                    ) or (
                        not self.params.greater_is_better
                        and metric_value < self.training_state.best_metric
                    ):
                        self.save_state(epoch + 1, self.params.output_dir)
                        self.training_state.best_metric = metric_value
                        self.training_state.best_epoch = epoch + 1
                        create_symlink(
                            self.params.output_dir / f"models/model_epoch_{epoch + 1}.pth",
                            self.params.output_dir / "models/model_best.pth",
                        )
                        self.logger.info(
                            "Save best model at "
                            f"{self.params.metric_for_best_model}"
                            f"({self.training_state.best_metric:.4f}) "
                            f"in Epoch {self.training_state.best_epoch}"
                        )
        self._after_train(self.training_state)

    def _before_epoch(self, state: TrainingState):
        self.optimizer.zero_grad(set_to_none=True)
        # For ScheduleFree Optimizer
        if state.phase == "train" and getattr(self.optimizer, "train", False):
            self.optimizer.train()
        elif state.phase == "val" and getattr(self.optimizer, "eval", False):
            self.optimizer.eval()

        progress_bar = enumerate(self.dataloaders[state.phase])
        if is_local_main_process():
            progress_bar = tqdm(
                progress_bar, total=len(self.dataloaders[state.phase]), dynamic_ncols=True
            )
        state.progress_bar = progress_bar

        self.model.train(state.phase == "train")
        model = self.model.module if is_distributed() else self.model
        model.phase = state.phase
        if is_distributed() and self.samplers[state.phase] is not None:
            self.samplers[state.phase].set_epoch(state.current_epoch)

    def _do_one_epoch(self, state: TrainingState) -> EpochResult:
        epoch_result = EpochResult()

        for iter_idx, inputs in state.progress_bar:
            with torch.set_grad_enabled(state.phase == "train"):
                # Prepare Input
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device, non_blocking=True)
                if self.batched_transforms[state.phase]:
                    inputs = self.batched_transforms[state.phase](inputs)

                # Forward
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.params.use_amp,
                    dtype=TORCH_DTYPE[self.params.amp_dtype],
                ):
                    outputs: ModelOutput = self.model(inputs)

                # Backward
                if state.phase == "train":
                    self._backward(outputs, iter_idx, state.current_epoch)

            self._after_forward(state, epoch_result, inputs, outputs)

        return epoch_result

    def _backward(self, output: ModelOutput, iter_idx: int, current_epoch: int):
        total_loss = output["losses"]["total_loss"] / self.params.gradient_accumulation_steps
        self.scaler.scale(total_loss).backward()

        if (iter_idx + 1) % self.params.gradient_accumulation_steps == 0:
            if self.params.use_clip_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.iter_lr_scheduler:
                self.iter_lr_scheduler.step(
                    epoch=iter_idx + (current_epoch - 1) * len(self.dataloaders["train"]),
                    metric=output["losses"]["total_loss"].item(),
                )

    def _after_forward(
        self,
        state: TrainingState,
        epoch_history: EpochResult,
        inputs: DatasetOutput,
        outputs: ModelOutput,
    ):
        if self.evaluators[state.phase] is not None:
            self.evaluators[state.phase].update(inputs, outputs["preds"])

        description = f"Epoch: {state.current_epoch}/{self.params.epochs}. "
        # Update Epoch Loss History
        for key, value in outputs["losses"].items():
            if is_distributed():
                value = reduce_tensor(value) / dist.get_world_size()
            epoch_history[key] = epoch_history.get(key, 0.0) + value.item()
            description += f"{key.capitalize()}: {value.item():8.4f} "
        if is_local_main_process():
            state.progress_bar.set_description(description)

    def _after_epoch(self, state: TrainingState, epoch_result: EpochResult) -> EpochResult:
        if state.phase == "train" and self.epoch_lr_scheduler:
            self.epoch_lr_scheduler.step(
                epoch=state.current_epoch, metric=epoch_result["total_loss"]
            )

        # Compute Epoch Result
        for k, v in epoch_result.items():
            epoch_result[k] = v / len(self.dataloaders[state.phase])
        if self.evaluators[state.phase] is not None:
            metrics: dict[str, Any] = self.evaluators[state.phase].compute()
            self.evaluators[state.phase].reset()
            epoch_result.update(metrics)
        if state.phase == "train":
            epoch_result["Learning Rate"] = self.optimizer.param_groups[0]["lr"]

        description = [
            f"{state.phase.capitalize()}",
            f"Epoch: {state.current_epoch}/{self.params.epochs}",
        ]
        if self.device.type == "cuda":
            description.append(f"GPU: {torch.cuda.memory_reserved(self.device) / 1e9:.1f}GB.")
        for k, v in epoch_result.items():
            if isinstance(v, float):
                description.append(f"{k}: {v:8.4f}")
            else:
                description.append(f"{k}: {v}")
        description = " ".join(description)
        logger.info(description)

        return epoch_result

    def _after_train(self, state: TrainingState):
        if is_world_main_process():
            self.logger.log_artifact(self.params.output_dir / "models/model_final.pth")
            self.logger.log_artifact(self.params.output_dir / "optimizers/optimizer_final.pth")
            if self.iter_lr_scheduler:
                self.logger.log_artifact(
                    self.params.output_dir / "schedulers/iter_scheduler_final.pth"
                )
            if self.epoch_lr_scheduler:
                self.logger.log_artifact(
                    self.params.output_dir / "schedulers/epoch_scheduler_final.pth"
                )
            self.logger.log_history_figure(self.params.output_dir / "graph")
            self.logger.log_artifacts(self.params.output_dir / "graph")

            best_epoch_path = (
                self.params.output_dir / f"models/model_epoch_{self.training_state.best_epoch}.pth"
            )
            best_model_path = self.params.output_dir / "models/model_best.pth"
            if best_epoch_path.exists():
                create_symlink(
                    best_epoch_path,
                    self.params.output_dir / "models/model_best.pth",
                )
                self.logger.log_artifact(best_model_path)
                self.cfg.model.checkpoint = best_model_path

            ConfigManager.dump(self.cfg, self.params.output_dir / "config.yaml")
            self.logger.log_artifact(self.params.output_dir / "config.yaml")

    def save_state(self, epoch: int, output_dir: Path) -> dict[str, Path]:
        weight_path = f"{output_dir}/models/model_epoch_{epoch}.pth"
        optimizer_path = f"{output_dir}/optimizers/optimizer_epoch_{epoch}.pth"
        epoch_scheduler_path = f"{output_dir}/schedulers/epoch_scheduler_epoch_{epoch}.pth"
        iter_scheduler_path = f"{output_dir}/schedulers/iter_scheduler_epoch_{epoch}.pth"

        final_model_path = f"{output_dir}/models/model_final.pth"
        final_optimizer_path = f"{output_dir}/optimizers/optimizer_final.pth"
        final_epoch_scheduler_path = f"{output_dir}/schedulers/epoch_scheduler_final.pth"
        final_iter_scheduler_path = f"{output_dir}/schedulers/iter_scheduler_final.pth"

        save_model(self.model, weight_path)

        if is_world_main_process():
            # Update Config
            self.cfg.last_epoch = epoch
            self.cfg.model.checkpoint = weight_path
            self.cfg.optimizer.checkpoint = optimizer_path
            if self.epoch_lr_scheduler:
                self.cfg.lr_scheduler.epoch_scheduler.checkpoint = epoch_scheduler_path
            if self.iter_lr_scheduler:
                self.cfg.lr_scheduler.iter_scheduler.checkpoint = iter_scheduler_path
            ConfigManager.dump(self.cfg, output_dir / "config.yaml")
            self.logger.log_artifact(output_dir / "config.yaml")

            create_symlink(weight_path, final_model_path)

            save_optimizer(self.optimizer, optimizer_path)
            create_symlink(optimizer_path, final_optimizer_path)

            if self.epoch_lr_scheduler:
                save_lr_scheduler(self.epoch_lr_scheduler, epoch_scheduler_path)
                create_symlink(epoch_scheduler_path, final_epoch_scheduler_path)

            if self.iter_lr_scheduler:
                save_lr_scheduler(self.iter_lr_scheduler, iter_scheduler_path)
                create_symlink(iter_scheduler_path, final_iter_scheduler_path)

    def load_state(
        self,
        start_epoch: int,
        best_metric: float,
        best_epoch: int,
        model_path: PathLike,
        optimizer_path: PathLike,
        epoch_scheduler_path: PathLike,
        iter_scheduler_path: PathLike,
    ):
        self.training_state.current_epoch = start_epoch
        self.training_state.best_metric = best_metric
        self.training_state.best_epoch = best_epoch

        load_model_weight(model_path, self.model)
        self.logger.info(f"Load model weight from {model_path}")

        # For ScheduleFree Optimizer
        if getattr(self.optimizer, "eval", None):
            self.optimizer.eval()
        self.optimizer.load_state_dict(
            torch.load(optimizer_path, map_location=self.device, weights_only=False)
        )
        self.logger.info(f"Load optimizer weight from {optimizer_path}")

        if self.epoch_lr_scheduler:
            self.epoch_lr_scheduler.load_state_dict(
                torch.load(epoch_scheduler_path, map_location=self.device, weights_only=False)
            )
            self.logger.info(f"Load epoch scheduler weight from {epoch_scheduler_path}")
        if self.iter_lr_scheduler:
            self.iter_lr_scheduler.load_state_dict(
                torch.load(iter_scheduler_path, map_location=self.device, weights_only=False)
            )
            self.logger.info(f"Load iter scheduler weight from {iter_scheduler_path}")


class Trainer(BaseTrainer):
    pass
