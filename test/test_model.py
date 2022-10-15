import sys
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf  # noqa

sys.path.append("../")
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

    model = build_model(cfg, device=torch.device("cpu"))

    x = torch.zeros((2, 3, 480, 480))

    y = model(x)

    print(y)


if __name__ == "__main__":
    main()
