"""Dataloaderを回して速度を計測"""

import os
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import ConfigManager, ExperimentConfig
from src.dataloaders import build_dataset


class TimeCompose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data: dict) -> dict:
        for t in self.transforms:
            start = time.time()
            data = t(data)
            print(f"    {t.__class__.__name__}: {time.time() - start:.4f}")
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"{t}, "
        format_string += ")"
        return format_string


@ConfigManager.argparse
def main(cfg: ExperimentConfig):
    cfg.batch = 2
    device = torch.device("cpu" if cfg.use_cpu else "cuda:0")

    for phase in ["train", "val", "test"]:
        print(f"Phase: {phase}")
        dataset, dataloader, batched_transforms = build_dataset(cfg, phase)
        dataloader.dataset.transform = TimeCompose(dataset.transforms.transforms)

        start_time = time.time()

        for i, data in enumerate(dataloader):
            start_one_batch = time.time()

            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device, non_blocking=True)
            if batched_transforms:
                data = batched_transforms(data)

            print(f"{i:4}/{len(dataloader)}: {((time.time() - start_one_batch) * 1000):.5f}ms")
            if i > 1:
                break

        print(f"{((time.time() - start_time) * 1000):.5f}ms")


if __name__ == "__main__":
    main()
