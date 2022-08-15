import os
import logging
import csv
import pprint
import pathlib

import torch

import numpy as np
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from clearml import Task

from kunai.torch_utils import (
    set_device,
    time_synchronized,
)
from kunai.hydra_utils import set_hydra, validate_config
from kunai.utils import get_cmd, get_git_hash, setup_logger

from src.models import build_model
from src.utils import (
    post_slack,
    make_result_dirs,
)
from train import build_dataset

# Get root logger
logger = logging.getLogger()


def do_test(cfg, output_dir, device):
    logger.info("Loading Dataset...")
    dataset, _ = build_dataset(cfg, phase="test")
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, num_workers=4, batch_size=1)

    logger.info(f"Load model weight {cfg.MODEL.WEIGHT}")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(cfg.MODEL.WEIGHT, map_location=device))
    logger.info("Complete load model")

    inference_speed = 0
    metric = 0
    results = []
    model.requires_grad_(False)
    model.eval()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        with torch.no_grad():
            input_data = data.to(device)
            t = time_synchronized()
            y = model(input_data)
            inference_speed += time_synchronized() - t

            # calc metrics below
            result = y
            results.append(result)

    inference_speed /= len(dataset)
    logger.info(
        f"Average Inferance Speed: {inference_speed:.5f}s, {(1.0 / inference_speed):.2f}fps"
    )

    # 評価結果の保存
    with open(os.path.join(output_dir, "result.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)
    return metric


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    set_hydra(cfg)
    cfg = validate_config(cfg)

    # Validate Model Weight Path
    config_path = os.path.basename(HydraConfig.get().runtime.config_sources[1].path)
    if config_path not in cfg.MODEL.WEIGHT:
        weight_dir_list = cfg.MODEL.WEIGHT.split("/")
        weight_dir_list[-3] = config_path
        cfg.MODEL.WEIGHT = os.path.join(*weight_dir_list)
        logger.info(f"Weight Path Changed {cfg.MODEL.WEIGHT}")

    output_dir = make_result_dirs(cfg.MODEL.WEIGHT)
    setup_logger(-1, os.path.join(output_dir, "test.log"))
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    logger.info(f"Command: {get_cmd()}")
    logger.info(f"Make output_dir at {output_dir}")
    logger.info(f"Git Hash: {get_git_hash()}")
    with open(
        os.path.join(os.path.dirname(os.path.dirname(output_dir)), "cmd_histry.log"), "a"
    ) as f:
        print(get_cmd(), file=f)

    # ClearML
    try:
        prefix = f"{cfg.MODEL.NAME}_{cfg.DATASET.NAME}"
        if cfg.TAG:
            prefix += f"_{cfg.TAG}"
        Task.init(project_name=pathlib.Path.cwd().name, task_name=prefix)
    except Exception:
        logger.info("Not Installed ClearML")

    # set Device
    device = set_device(cfg.GPU.USE, is_cpu=cfg.CPU)

    result = do_test(cfg, output_dir, device)

    message = pprint.pformat(
        {
            "host": os.uname()[1],
            "tag": cfg.TAG,
            "model": cfg.MODEL.NAME,
            "dataset": cfg.DATASET.NAME,
            "save": output_dir,
            "result": f"{result:7.3f}",
        },
        width=150,
    )
    # Send Message to Slack
    post_slack(message=f"Finish Test\n{message}")
    logger.info(f"Finish Test {message}")

    return result


if __name__ == "__main__":
    main()
