import logging
import random
import sys

import numpy as np
import torch

sys.path.append("./")
from src.losses import build_loss
from src.models import build_model  # noqa
from src.utils import Config

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


@Config.main
def main(cfg):
    logger = logging.getLogger()
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    logger.setLevel(logging.INFO)
    default_handler = logger.handlers[0]
    default_handler.setLevel(logging.INFO)

    # cfg.TRANSFORMS = OmegaConf.create(transform)
    cfg.MODEL.PRE_TRAINED = False
    model, _ = build_model(cfg, device=torch.device("cpu"), phase="train")
    criterion = build_loss(cfg)

    # print(model)

    x = torch.zeros((2, 3, *cfg.MODEL.INPUT_SIZE))
    box = torch.randint(0, 100, (2, 8, 4)).float()
    box[:, :, 2:] += 100
    data = {
        "bbox": box,
        "class": torch.randint(0, 20, (2, 8, 1)).long(),
        "scale": torch.ones(2),
        "image_size": torch.tensor([cfg.MODEL.INPUT_SIZE, cfg.MODEL.INPUT_SIZE]),
        "org_size": torch.tensor([cfg.MODEL.INPUT_SIZE, cfg.MODEL.INPUT_SIZE]),
        "mask": torch.ones((2, *cfg.MODEL.INPUT_SIZE)).bool(),
    }
    # model.phase = "val"

    y = model(x, data)
    # print(y)

    loss = criterion(y, data)
    print(loss)

    # print(y)


if __name__ == "__main__":
    main()
