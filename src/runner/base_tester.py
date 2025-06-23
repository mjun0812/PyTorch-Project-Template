from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

from ..models import BaseModel, ModelOutput
from ..types import TORCH_DTYPE
from ..utils import time_synchronized


@dataclass
class TesterOutput:
    """Output container for test results.

    Attributes:
        results: List of model prediction dictionaries for each sample.
        targets: List of ground truth target dictionaries for each sample.
        inference_speed: Average inference time per sample in seconds.
        metrics: Dictionary of computed evaluation metrics, if available.
    """

    results: list[dict[str, Any]] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    inference_speed: float = 0.0
    metrics: dict[str, Any] | None = None


class BaseTester:
    """Base class for model testing and evaluation.

    Provides a standardized interface for running inference on test datasets,
    collecting results, computing metrics, and measuring inference performance.

    Attributes:
        device: PyTorch device for computation.
        model: Model to evaluate.
        dataloader: DataLoader for test data.
        batched_transform: Optional batch-level transformation function.
        evaluator: Optional metric collection for evaluation.
        use_amp: Whether to use automatic mixed precision.
        amp_dtype: Data type for AMP operations.
    """

    def __init__(
        self,
        device: torch.device,
        model: BaseModel,
        dataloader: DataLoader,
        batched_transform: Callable | None = None,
        evaluator: MetricCollection | None = None,
        use_amp: bool = True,
        amp_dtype: str = "fp32",
    ) -> None:
        """Initialize the base tester.

        Args:
            device: PyTorch device for computation.
            model: Model to evaluate.
            dataloader: DataLoader providing test data.
            batched_transform: Optional function to apply batch-level transformations.
            evaluator: Optional metric collection for computing evaluation metrics.
            use_amp: Whether to enable automatic mixed precision.
            amp_dtype: Data type for AMP (fp16, bf16, or fp32).
        """
        self.device = device
        self.model = model
        self.model.phase = "test"
        self.dataloader = dataloader
        self.batched_transform = batched_transform
        self.evaluator = evaluator
        self.use_amp = use_amp
        self.amp_dtype = TORCH_DTYPE[amp_dtype] if use_amp else torch.float32

    def do_test(self) -> TesterOutput:
        """Execute the complete testing process.

        Runs inference on all test data, collects results and targets,
        computes metrics, and measures inference performance.

        Returns:
            TesterOutput containing results, targets, metrics, and performance data.
        """
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

    def _setup_pbar(self) -> tqdm:
        """Set up progress bar for testing loop.

        Returns:
            Progress bar instance for tracking test progress.
        """
        progress_bar = enumerate(self.dataloader)
        progress_bar = tqdm(progress_bar, total=len(self.dataloader), dynamic_ncols=True)
        return progress_bar

    def prepare_input(self, data: dict) -> dict:
        """Prepare input data for model inference.

        Moves tensors to the appropriate device and applies batch transformations.

        Args:
            data: Input data dictionary from the dataloader.

        Returns:
            Processed data dictionary ready for model input.
        """
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device, non_blocking=True)
        if self.batched_transform:
            data = self.batched_transform(data)
        return data

    def update_metrics(self, data: dict, preds: dict) -> None:
        """Update evaluation metrics with new predictions and targets.

        Args:
            data: Ground truth data dictionary.
            preds: Model predictions dictionary.
        """
        if self.evaluator is not None:
            self.evaluator.update(*self.generate_input_evaluator(data, preds))

    def collect_results(
        self, data: dict, output: ModelOutput, results: list, targets: list
    ) -> tuple[list, list]:
        """Collect and store results and targets from a batch.

        Moves tensors to CPU and appends them to the result collections.

        Args:
            data: Ground truth data dictionary for the batch.
            output: Model output containing predictions.
            results: List to append predictions to.
            targets: List to append ground truth data to.

        Returns:
            Tuple of updated (results, targets) lists.
        """
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
        """Process results after testing completion.

        Computes final metrics, calculates average inference speed,
        and organizes results into the output format.

        Args:
            results: List of prediction dictionaries from all batches.
            targets: List of target dictionaries from all batches.
            inference_times: List of inference times for each batch.

        Returns:
            TesterOutput containing organized results and computed metrics.
        """
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
        """Extract individual samples from batched data.

        Converts list of batch dictionaries to list of individual sample dictionaries.

        Args:
            data: List of batch dictionaries.

        Returns:
            List of individual sample dictionaries.
        """
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

    def generate_input_evaluator(self, targets: Any, preds: Any) -> tuple[Any, Any]:
        """Generate input format for the evaluator.

        Can be overridden to customize how targets and predictions
        are formatted for specific evaluation metrics.

        Args:
            targets: Ground truth targets.
            preds: Model predictions.

        Returns:
            Tuple of (targets, preds) formatted for the evaluator.
        """
        return targets, preds

    def save_results(self, output_dir: Path, targets: list, results: list) -> None:
        """Save test results to files.

        Base implementation does nothing. Override in subclasses to implement
        custom result saving logic.

        Args:
            output_dir: Directory to save results to.
            targets: List of ground truth targets.
            results: List of model predictions.
        """
        pass
