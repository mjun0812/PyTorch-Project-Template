from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from loguru import logger
from torchmetrics import MetricCollection
from tqdm import tqdm

from .models import BaseModel, ModelOutput
from .utils import TORCH_DTYPE, time_synchronized


@dataclass
class TesterOutput:
    results: list[dict] = field(default_factory=list)
    targets: list[dict] = field(default_factory=list)
    inference_speed: float = 0.0


class Tester:
    def __init__(
        self,
        device: torch.device,
        model: BaseModel,
        dataloader: torch.utils.data.DataLoader,
        batched_transform: callable,
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
        self.amp_dtype = TORCH_DTYPE[amp_dtype] if use_amp else torch.float32

        self.use_amp = use_amp
        if self.use_amp:
            logger.info("Using Mixed Precision with AMP")
            logger.info(f"AMP dtype: {self.amp_dtype}")

    def do_test(self):
        progress_bar = tqdm(
            enumerate(self.dataloader), total=len(self.dataloader), dynamic_ncols=True
        )

        results = []
        targets = []
        inference_times = []

        for i, data in progress_bar:
            with torch.inference_mode():
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device, non_blocking=True)
                if self.batched_transform:
                    data = self.batched_transform(data)

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):
                    t = time_synchronized()
                    output: ModelOutput = self.model(data)
                    inference_times.append(time_synchronized() - t)

            if self.evaluator is not None:
                self.evaluator.update(*self.generate_input_evaluator(data, output["preds"]))

            for k, v in output["preds"].items():
                if isinstance(v, torch.Tensor):
                    output["preds"][k] = v.cpu().detach()
            results.append(output["preds"])
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cpu().detach()
            targets.append(data)

        inference_speed = np.mean(inference_times[len(inference_times) // 2 :])
        results = self._extract_batch_list(results)
        targets = self._extract_batch_list(targets)
        return TesterOutput(results=results, targets=targets, inference_speed=inference_speed)

    def _extract_batch_list(self, data: dict) -> list[dict]:
        batch_list = []
        for d in data:
            batch_size = 1
            for v in d.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.size(0)

            for i in range(batch_size):
                batch_list.append({k: v[i] for k, v in d.items()})
        return batch_list

    def generate_input_evaluator(self, targets, preds):
        return targets, preds

    def save_results(self, output_dir, targets, results):
        pass
