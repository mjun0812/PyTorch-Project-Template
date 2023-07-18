import csv
import json
import logging
import os
import pprint
from pathlib import Path

import numpy as np
import torch
from kunai.torch_utils import set_device, time_synchronized
from kunai.utils import get_cmd, get_git_hash, setup_logger
from tqdm import tqdm

from src.dataloaders import build_dataset
from src.models import build_model
from src.utils import Config, Writer, make_result_dirs, post_slack

# Get root logger
logger = logging.getLogger()


def do_test(cfg, output_dir, device, writer: Writer):
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
    inference_times = []
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True)
    for i, data in progress_bar:
        with torch.no_grad():
            input_data = data.to(device)

            t = time_synchronized()
            y = model(input_data)
            inference_times.append(time_synchronized() - t)

            # calc metrics below
            result = y
            results.append(result)

    inference_speed = np.mean(inference_times[len(inference_times) // 2 :])
    logger.info(
        f"Average Inferance Speed: {inference_speed:.5f}s, {(1.0 / inference_speed):.2f}fps"
    )

    # 評価結果の保存
    with open(os.path.join(output_dir, "result.csv"), "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(results)

    metric_dict = {"result": metric, "Speed/s": inference_speed, "fps": 1.0 / inference_speed}
    for name, value in metric_dict.items():
        logger.info(f"{name}: {value}")
        writer.log_metric(name, value, None)
    json.dump(metric_dict, open(os.path.join(output_dir, "result.json"), "w"), indent=2)
    return metric


@Config.main
def main(cfg):
    # set Device
    device = set_device(cfg.GPU.USE, use_cudnn=cfg.CUDNN, is_cpu=cfg.CPU)

    output_dir = make_result_dirs(cfg.MODEL.WEIGHT)

    setup_logger(os.path.join(output_dir, "test.log"))
    logger.info(f"Command: {get_cmd()}")
    logger.info(f"Make output_dir at {output_dir}")
    logger.info(f"Git Hash: {get_git_hash()}")
    with open(Path(output_dir).parents[1] / "cmd_histry.log", "a") as f:
        print(get_cmd(), file=f)
    Config.dump(cfg, os.path.join(output_dir, "config.yaml"))

    writer = Writer(cfg, output_dir, "test")
    writer.log_artifact(Path(output_dir).parents[1] / "cmd_histry.log")
    writer.log_artifact(os.path.join(output_dir, "config.yaml"))

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
