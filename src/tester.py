import numpy as np
import torch
from tqdm import tqdm

from .utils import TORCH_DTYPE, time_synchronized


class Tester:
    def __init__(
        self,
        cfg,
        device: torch.device,
        model,
        dataloader,
        batched_transform,
        evaluator,
        use_amp,
        amp_dtype="fp32",
    ):
        self.cfg = cfg
        self.device = device
        self.model = model
        self.model.phase = "test"
        self.dataloader = dataloader
        self.batched_transform = batched_transform
        self.evaluator = evaluator
        self.use_amp = use_amp
        self.amp_dtype = TORCH_DTYPE[amp_dtype] if use_amp else torch.float32
        self.is_cpu = device.type == "cpu"

    def do_test(self):
        progress_bar = tqdm(
            enumerate(self.dataloader), total=len(self.dataloader), dynamic_ncols=True
        )

        results = []
        targets = []
        inference_times = []

        for i, (image, data) in progress_bar:
            with torch.no_grad():
                image = image.to(self.device).float()
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device, non_blocking=True)
                if self.batched_transform:
                    image, data = self.batched_transform(image, data)

                with torch.autocast(
                    device_type="cuda" if not self.is_cpu else "cpu",
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):
                    t = time_synchronized()
                    output = self.model(image, data)
                    inference_times.append(time_synchronized() - t)

                self.evaluator.update(*self.generate_input_evaluator(output, data))

                for k, v in output.items():
                    if isinstance(v, torch.Tensor):
                        output[k] = v.cpu().detach()
                results.append(output)
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.cpu()
                targets.append(data)

        inference_speed = np.mean(inference_times[len(inference_times) // 2 :])
        return {"outputs": results, "targets": targets, "inference_speed": inference_speed}

    def generate_input_evaluator(self, output, data):
        return output, data

    def save_results(self, output_dir, results, targets):
        pass
