"""Datasetを読み込めるかテスト"""

import sys

import hydra
from omegaconf import DictConfig  # noqa
from tqdm import tqdm

sys.path.append("./")
from src.dataloaders import build_dataset  # noqa


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    data = build_dataset(cfg, "train")
    if len(data) == 2:
        dataset, dataloader = data
    else:
        dataset, dataloader, batched_transforms = data
    print("Loading dataset Complete")

    for i, data in enumerate(tqdm(dataloader)):
        image, data = data

        if i == min(100, len(dataloader) - 1):
            break


if __name__ == "__main__":
    main()
