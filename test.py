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
from src.utils import Config, Writer, build_evaluator, make_result_dirs, post_slack

# Get root logger
logger = logging.getLogger()


class Tester:
    def __init__(
        self,
        cfg,
        device: torch.device,
        model,
        dataloader,
        evaluator,
    ):
        self.cfg = cfg
        self.device = device
        self.model = model
        self.model.phase = "test"
        self.dataloader = dataloader
        self.evaluator = evaluator

    def do_test(self):
        progress_bar = tqdm(
            enumerate(self.dataloader), total=len(self.dataloader), dynamic_ncols=True
        )

        results = []
        targets = []
        inference_times = []

        for i, (image, data) in progress_bar:
            with torch.no_grad():
                image = image.to(self.device)
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device, non_blocking=True)

                t = time_synchronized()
                output = self.model(image, data)
                inference_times.append(time_synchronized() - t)

                self.evaluator.update(*self.generate_input_evaluator(output, data))

                results.append(output)
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.cpu()
                targets.append(data)

        inference_speed = np.mean(inference_times[len(inference_times) // 2 :])
        return {"outputs": results, "targets": targets, "inference_speed": inference_speed}

    def generate_input_evaluator(self, output, data):
        return output, data

    def save_results(self, output_dir, results, targets):
        pass


def do_test(cfg, output_dir, device, writer: Writer):
    logger.info("Loading Dataset...")
    dataset, _, _ = build_dataset(cfg, phase="test")
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, num_workers=4, batch_size=1)

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

    evaluator = build_evaluator(cfg, phase="train").to(device)

    tester = Tester(cfg, device, model, dataloader, evaluator)
    results = tester.do_test()

    inference_speed = results["inference_speed"]
    logger.info(
        f"Average Inferance Speed: {inference_speed:.5f}s, " f"{(1.0 / inference_speed):.2f}fps"
    )

    tester.save_results(output_dir, results["outputs"], results["targets"])

    metrics = evaluator.compute()
    metrics.update(
        {
            "Speed/s": inference_speed,
            "fps": 1.0 / inference_speed,
        }
    )
    for name, value in metrics.items():
        logger.info(f"{name}: {value}")
    writer.log_metrics(metrics, None)
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(metrics, f, indent=2)


@Config.main
def main(cfg):
    # set Device
    device = set_device(cfg.GPU.USE, use_cudnn=cfg.CUDNN, is_cpu=cfg.CPU)

    output_dir = Path(make_result_dirs(cfg.MODEL.WEIGHT))

    setup_logger(output_dir / "test.log")
    logger.info(f"Command: {get_cmd()}")
    logger.info(f"Make output_dir at {str(output_dir)}")
    logger.info(f"Git Hash: {get_git_hash()}")
    with open(output_dir.parents[1] / "cmd_histry.log", "a") as f:
        print(get_cmd(), file=f)
    Config.dump(cfg, output_dir / "config.yaml")

    writer = Writer(cfg, output_dir, "test")
    writer.log_artifact(output_dir.parents[1] / "cmd_histry.log")
    writer.log_artifact(output_dir, "config.yaml")

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
