"""Datasetを読み込めるかテスト"""

import sys

from tqdm import tqdm

sys.path.append("./")
from src.dataloaders import build_dataset  # noqa
from src.utils.config import Config


@Config.main
def main(cfg):
    dataset, dataloader, batched_transforms = build_dataset(cfg, "train")
    print("Loading dataset Complete")

    for i, data in enumerate(tqdm(dataloader)):
        image, data = data

        if i == min(100, len(dataloader) - 1):
            break


if __name__ == "__main__":
    main()
