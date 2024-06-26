import logging
import os
import random
import sys

import numpy as np
import torch

sys.path.append("./")
from src.dataloaders import build_dataset  # noqa
from src.losses import build_loss  # noqa
from src.models import build_model  # noqa
from src.utils import Config  # noqa

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


@Config.main
def main(cfg):
    phase = cfg.get("PHASE", "train")
    print(f"Phase: {phase}")
    logger = logging.getLogger()
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    logger.setLevel(logging.INFO)
    default_handler = logger.handlers[0]
    default_handler.setLevel(logging.INFO)

    cfg.BATCH = 1
    cfg.MODEL.PRE_TRAINED = False

    model, _ = build_model(cfg, device=torch.device("cpu"), phase=phase)
    criterion = build_loss(cfg)
    print(model)

    _, dataloader, batched_transforms = build_dataset(cfg, phase)
    print("Loading dataset Complete")

    data = next(iter(dataloader))
    if batched_transforms:
        data = batched_transforms(data)
    image = data["image"][0]
    print(image.shape)

    y = model(data)
    print(y)
    if phase != "test":
        loss = criterion(y, data)
        print(loss)


if __name__ == "__main__":
    main()
