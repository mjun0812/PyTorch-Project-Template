"""Dataloaderを回して速度を計測"""

import sys
import time

import torch

sys.path.append("./")
from src.dataloaders import build_dataset  # noqa
from src.utils.config import Config


class TimeCompose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, annotations: dict):
        for t in self.transforms:
            start = time.time()
            img, annotations = t(img, annotations)
            print(f"{t.__class__.__name__}: {time.time()-start:.4f}")
        return img, annotations

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"{t}, "
        format_string += ")"
        return format_string


@Config.main
def main(cfg):
    dataset, dataloader, batched_transforms = build_dataset(cfg, "train")
    dataloader.dataset.transform = TimeCompose(dataset.transform.transforms)
    print("Loading dataset Complete")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    loader_time = time.time()
    start_time = loader_time

    for _ in range(2):
        for i, data in enumerate(dataloader):
            image, data = data
            image = image.to(device, non_blocking=True).float()
            tmp = time.time()
            print(f"{i:4}/{len(dataloader):5}: {tmp - start_time:.5f}")
            start_time = tmp

    print(f"{time.time() - loader_time:.5f}")


if __name__ == "__main__":
    main()
