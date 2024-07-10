import json
import os
import pprint
from pathlib import Path

import torch

from src.dataloaders import build_dataset
from src.models import build_model
from src.tester import Tester
from src.utils import (
    Config,
    JsonEncoder,
    Logger,
    build_evaluator,
    get_cmd,
    load_model_weight,
    make_result_dirs,
    post_slack,
    set_device,
)


def do_test(cfg, output_dir, device, logger: Logger):
    logger.info("Loading Dataset...")
    _, dataloader, batched_transform = build_dataset(cfg, phase="test", logger=logger)

    model, _ = build_model(cfg, device, phase="test", logger=logger)
    load_model_weight(cfg.MODEL.WEIGHT, model, logger)
    model.eval()
    model.requires_grad_(False)
    logger.info("Complete load model")

    evaluator = build_evaluator(cfg, phase="train")
    if evaluator is not None:
        evaluator = evaluator.to(device)

    if cfg.AMP:
        logger.info("Using Mixed Precision with AMP")
        logger.info(f"AMP dtype: {cfg.AMP_DTYPE}")

    tester = Tester(
        cfg,
        device,
        model,
        dataloader,
        batched_transform,
        evaluator,
        use_amp=cfg.AMP,
        amp_dtype=cfg.AMP_DTYPE,
    )
    results = tester.do_test()

    inference_speed = results["inference_speed"]
    logger.info(
        f"Average Inferance Speed: {inference_speed:.5f}s, {(1.0 / inference_speed):.2f}fps"
    )

    tester.save_results(output_dir, results["outputs"], results["targets"])

    metrics = {}
    if evaluator is not None:
        metrics = evaluator.compute()
    metrics.update(
        {
            "Speed/s": inference_speed,
            "fps": 1.0 / inference_speed,
        }
    )
    for name, value in metrics.items():
        logger.info(f"{name}: {value}")
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            metrics[name] = value.item()
    logger.log_metrics(metrics, None)
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(metrics, f, indent=2, cls=JsonEncoder)


@Config.main
def main(cfg):
    # set Device
    device = set_device(cfg.GPU.USE, use_cudnn=cfg.CUDNN, is_cpu=cfg.CPU)

    output_dir = Path(make_result_dirs(cfg.MODEL.WEIGHT))

    Config.dump(cfg, output_dir / "config.yaml")
    logger = Logger(str(output_dir), str(output_dir / "test.log"), "test")
    if cfg.USE_MLFLOW:
        logger.setup_mlflow(str(output_dir.parents[1].name), cfg.MLFLOW_EXPERIMENT_NAME)
        logger.log_config(cfg, cfg.MLFLOW_LOG_CONGIG_PARAMS)
    logger.info("\n" + Config.pretty_text(cfg))
    logger.log_artifact(output_dir.parents[1] / "cmd_histry.log")
    logger.log_artifact(output_dir / "config.yaml")
    with open(output_dir.parents[1] / "cmd_histry.log", "a") as f:
        print(get_cmd(), file=f)

    result = do_test(cfg, output_dir, device, logger)

    message = pprint.pformat(
        {
            "host": os.uname()[1],
            "tag": cfg.TAG,
            "model": cfg.MODEL.NAME,
            "dataset": cfg.TEST_DATASET.NAME,
            "save": str(output_dir),
            "result": f"{result:7.3f}",
        },
        width=150,
    )
    # Send Message to Slack
    post_slack(message=f"Finish Test\n{message}")
    logger.info(f"Finish Test {message}")
    logger.log_result_dir(str(output_dir), ignore_dirs=cfg.MLFLOW_IGNORE_DIRS)
    logger.close()

    return result


if __name__ == "__main__":
    main()
