from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

from .models import BaseModel
from .models.types import ModelOutput
from .types import TORCH_DTYPE
from .utils import time_synchronized


@dataclass
class TesterOutput:
    results: list[dict[str, Any]] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    inference_speed: float = 0.0
    metrics: dict[str, Any] | None = None


class BaseTester:
    def __init__(
        self,
        device: torch.device,
        model: BaseModel,
        dataloader: DataLoader,
        batched_transform: callable | None = None,
        evaluator: MetricCollection | None = None,
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
            batch_size = 0
            for v in d.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.size(0)
                    break
            for batch_idx in range(batch_size):
                batch = {k: v[batch_idx] for k, v in d.items()}
                batch_list.append(batch)
        return batch_list

    def generate_input_evaluator(self, targets, preds):
        return targets, preds

    def save_results(self, output_dir, targets, results):
        pass


class Tester(BaseTester):
    pass
