from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

from .models import BaseModel, ModelOutput
from .utils import TORCH_DTYPE, time_synchronized


@dataclass
class TesterOutput:
    results: List[Dict[str, Any]] = field(default_factory=list)
    targets: List[Dict[str, Any]] = field(default_factory=list)
    inference_speed: float = 0.0
    metrics: Optional[Dict[str, Any]] = None


class BaseTester:
    def __init__(
        self,
        device: torch.device,
        model: BaseModel,
        dataloader: DataLoader,
        batched_transform: Optional[callable] = None,
        evaluator: Optional[MetricCollection] = None,
        use_amp: bool = True,
        amp_dtype: str = "fp32",
    ):
        self.device = device
        self.model = model
        self.model.phase = "test"
        self.dataloader = dataloader
        self.batched_transform = batched_transform
        self.evaluator = evaluator
        self.use_amp = use_amp
        self.amp_dtype = TORCH_DTYPE[amp_dtype] if use_amp else torch.float32

        if self.use_amp:
            logger.info(f"Using Mixed Precision with AMP (dtype: {self.amp_dtype})")

    def do_test(self) -> TesterOutput:
        results = []
        targets = []
        inference_times = []

        pbar = self._setup_pbar()

        for i, data in pbar:
            with torch.inference_mode():
                data = self.prepare_input(data)
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                    cache_enabled=False,
                ):
                    t = time_synchronized()
                    output: ModelOutput = self.model(data)
                    inference_time = time_synchronized() - t

            self.update_metrics(data, output["preds"])
            results, targets = self.collect_results(data, output, results, targets)
            inference_times.append(inference_time)

        return self.after_test(results, targets, inference_times)

    def _setup_pbar(self):
        progress_bar = enumerate(self.dataloader)
        progress_bar = tqdm(progress_bar, total=len(self.dataloader), dynamic_ncols=True)
        return progress_bar

    def prepare_input(self, data: dict) -> dict:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device, non_blocking=True)
        if self.batched_transform:
            data = self.batched_transform(data)
        return data

    def update_metrics(self, data: dict, preds: dict):
        if self.evaluator is not None:
            self.evaluator.update(*self.generate_input_evaluator(data, preds))

    def collect_results(
        self, data: dict, output: ModelOutput, results: list, targets: list
    ) -> tuple[list, list]:
        for k, v in output["preds"].items():
            if isinstance(v, torch.Tensor):
                output["preds"][k] = v.cpu().detach()
        results.append(output["preds"])
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cpu().detach()
        targets.append(data)
        return results, targets

    def after_test(self, results: list, targets: list, inference_times: list) -> TesterOutput:
        inference_speed = np.mean(inference_times[len(inference_times) // 2 :])
        results = self._extract_batch_list(results)
        targets = self._extract_batch_list(targets)
        metrics = {}
        if self.evaluator:
            metrics = self.evaluator.compute()
            self.evaluator.reset()
        return TesterOutput(
            results=results, targets=targets, inference_speed=inference_speed, metrics=metrics
        )

    def _extract_batch_list(self, data: list[dict]) -> list[dict]:
        batch_list = []
        for d in data:
            batch_size = next((v.size(0) for v in d.values() if isinstance(v, torch.Tensor)), 1)
            batch_list.extend(
                [
                    {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                    for i in range(batch_size)
                ]
            )
        return batch_list

    def generate_input_evaluator(self, targets, preds):
        return targets, preds

    def save_results(self, output_dir, targets, results):
        pass


class Tester(BaseTester):
    def default_generate_input_evaluator(self, targets, preds):
        return targets, preds

    def save_results(self, output_dir, targets, results):
        return super().save_results(output_dir, targets, results)
