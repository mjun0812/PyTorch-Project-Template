import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import ConfigManager, ExperimentConfig
from src.dataloaders import build_dataset
from src.models import build_model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


@ConfigManager.argparse
def main(cfg: ExperimentConfig):
    cfg.model.pre_trained_weight = None
    model = build_model(cfg, device=torch.device("cpu" if cfg.use_cpu else "cuda:0"), phase="train")
    model.eval()
    model.requires_grad_(False)
    print(model)

    for phase in ["train", "val", "test"]:
        print(f"Phase: {phase}")
        model.phase = phase

        _, dataloader, batched_transforms = build_dataset(cfg, phase)

        data = next(iter(dataloader))
        if batched_transforms:
            data = batched_transforms(data)
        print(f"Data: {data}")
        y = model(data)
        print(f"Output: {y}")


if __name__ == "__main__":
    main()
