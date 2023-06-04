import logging
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf  # noqa

sys.path.append("./")
from src.losses import build_loss
from src.models import build_model  # noqa


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger()
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    logger.setLevel(logging.INFO)
    default_handler = logger.handlers[0]
    default_handler.setLevel(logging.INFO)

    # cfg.TRANSFORMS = OmegaConf.create(transform)
    cfg.MODEL.PRE_TRAINED = False
    print(type(cfg.MODEL.INPUT_SIZE))
    model, _ = build_model(cfg, device=torch.device("cpu"), phase="val")
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_parameters)

    print(model)

    x = torch.zeros((2, 3, 512, 512))
    data = {
        "bbox": torch.zeros((2, 8, 4)).float(),
        "class": torch.zeros((2, 8, 1)).squeeze(-1).long(),
        "scale": torch.ones(2),
        "image_size": torch.tensor([[512, 512], [512, 512]]),
        "org_size": torch.tensor([[512, 512], [512, 512]]),
        "mask": torch.zeros((2, 512, 512)).bool(),
    }
    # model.phase = "val"

    y = model(x, data)
    print(y)

    # print(y)


if __name__ == "__main__":
    main()
