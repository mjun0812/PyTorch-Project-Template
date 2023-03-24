import os
import logging
import csv
import pprint
import json
from pathlib import Path

import torch

import numpy as np
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from kunai.torch_utils import set_device, time_synchronized
from kunai.hydra_utils import set_hydra, validate_config
from kunai.utils import get_cmd, get_git_hash, setup_logger

from src.models import build_model
from src.utils import post_slack, make_result_dirs, TestLogger
from src.dataloaders import build_dataset

# Get root logger
logger = logging.getLogger()


def calc_inference_speed(model, dataset):
    inference_speed = 0
    num_iter = 50
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, num_workers=4, batch_size=1)
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            t = time_synchronized()
            _ = model(data)
            inference_speed += time_synchronized() - t
            if i == num_iter:
                break
    return inference_speed / num_iter


def do_test(cfg, output_dir, device, writer):
    logger.info("Loading Dataset...")
    dataset, _ = build_dataset(cfg, phase="test")
    dataloader = torch.utils.data.DataLoader(
        dataset, pin_memory=True, num_workers=4, batch_size=cfg.BATCH
    )

    model, _ = build_model(cfg, device, phase="test")
    check_point = torch.load(cfg.MODEL.WEIGHT, map_location=device)
    # for torch.compile model
    state_dict = {}
    for k, v in check_point.items():
        state_dict[k.replace("_orig_mod.", "").replace("module.", "")] = v
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()
    logger.info(f"Load model weight {cfg.MODEL.WEIGHT}")
    logger.info("Complete load model")

    metric = 0
    results = []
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True)
    for i, data in progress_bar:
        with torch.no_grad():
            input_data = data.to(device)
            y = model(input_data)
            # calc metrics below
            result = y
            results.append(result)

    inference_speed = calc_inference_speed(model, dataset)
    logger.info(
        f"Average Inferance Speed: {inference_speed:.5f}s, {(1.0 / inference_speed):.2f}fps"
    )

    # 評価結果の保存
    with open(os.path.join(output_dir, "result.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    metric_dict = {"result": metric, "Speed/s": inference_speed, "fps": 1.0 / inference_speed}
    for name, value in metric_dict.items():
        logger.info(f"{name}: {value}")
        writer.log_metric(name, value, "test", None)
    json.dump(metric_dict, open(os.path.join(output_dir, "result.json"), "w"), indent=2)
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

    # set Device
    device = set_device(cfg.GPU.USE, is_cpu=cfg.CPU)

    output_dir = make_result_dirs(cfg.MODEL.WEIGHT)

    setup_logger(os.path.join(output_dir, "test.log"))
    logger.info(f"Command: {get_cmd()}")
    logger.info(f"Make output_dir at {output_dir}")
    logger.info(f"Git Hash: {get_git_hash()}")
    with open(Path(output_dir).parents[1] / "cmd_histry.log", "a") as f:
        print(get_cmd(), file=f)
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    # Set Tensorboard, MLflow
    writer = TestLogger(cfg, output_dir, str(Path(output_dir).parents[2]))
    writer.log_artifact(Path(output_dir).parents[1] / "cmd_histry.log")
    writer.log_artifact(os.path.join(output_dir, "config.yaml"))
    writer.log_tag("model_weight_test", cfg.MODEL.WEIGHT)

    result = do_test(cfg, output_dir, device, writer)

    writer.log_result_dir(output_dir)
    writer.close()

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
