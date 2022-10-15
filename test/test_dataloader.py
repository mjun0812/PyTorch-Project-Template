import sys
import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf  # noqa

sys.path.append("../")
from src.dataloaders import build_dataset  # noqa


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    dataset, dataloader = build_dataset(cfg, "train")
    # dataset, dataloader, batched_transforms = build_dataset(cfg, "train")
    print("Loading dataset Complete")
    device = torch.device("cuda:0")

    loader_time = time.time()
    start_time = loader_time

    for _ in range(2):
        for i, data in enumerate(dataloader):
            image, target, data = data
            image = image.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            tmp = time.time()
            print(f"{i:4}/{len(dataloader):5}: {tmp - start_time:.5f}")
            start_time = tmp

    print(f"{time.time() - loader_time:.5f}")


if __name__ == "__main__":
    main()
