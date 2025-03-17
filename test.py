import json
import os
from pathlib import Path

import torch

from src.config import ConfigManager, ExperimentConfig
from src.dataloaders import build_dataset
from src.evaluator import build_evaluator
from src.models import build_model
from src.tester import Tester
from src.utils import (
    JsonEncoder,
    Logger,
    get_cmd,
    load_model_weight,
    make_result_dirs,
    post_slack,
    set_device,
)


def do_test(cfg: ExperimentConfig, output_dir: Path, device: torch.device, logger: Logger):
    logger.info("Loading Dataset...")
    _, dataloader, batched_transform, _ = build_dataset(cfg, phase="test")
    logger.info("Complete Load Dataset")

    logger.info("Building Model...")
    model = build_model(cfg, device, "test")
    load_model_weight(cfg.model.trained_weight, model)
    model.eval()
    model.requires_grad_(False)
    logger.info("Complete Build model")

    logger.info("Building Evaluator...")
    evaluator = build_evaluator(cfg, "test")
    if evaluator is not None:
        evaluator = evaluator.to(device)
    logger.info("Complete Build Evaluator")

    tester = Tester(
        device,
        model,
        dataloader,
        batched_transform,
        evaluator,
        use_amp=cfg.use_amp,
        amp_dtype=cfg.amp_dtype,
    )
    results = tester.do_test()

    inference_speed = results.inference_speed
    logger.info(f"Speed/ms: {inference_speed * 1000:.5f}ms")
    logger.info(f"FPS: {1.0 / inference_speed:.2f}")
    metrics = results.metrics
    metrics.update({"Speed/ms": inference_speed * 1000, "fps": 1.0 / inference_speed})
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            metrics[name] = value.item()
    logger.log_metrics(metrics, None, "test")
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(metrics, f, indent=2, cls=JsonEncoder)

    tester.save_results(output_dir, results.targets, results.results)


@ConfigManager.argparse
def main(cfg: ExperimentConfig):
    # set Device
    device = set_device(cfg.gpu.use, use_cudnn=cfg.gpu.use_cudnn, is_cpu=cfg.use_cpu)

    base_dir = Path(cfg.model.trained_weight).parents[1]
    output_dir = make_result_dirs(base_dir)

    ConfigManager.dump(cfg, output_dir / "config.yaml")
    logger = Logger(
        output_dir,
        "test",
        "INFO",
        use_mlflow=cfg.mlflow.use,
        use_wandb=cfg.wandb.use,
        mlflow_experiment_name=cfg.mlflow.experiment_name,
        wandb_project_name=cfg.wandb.project_name,
    )
    logger.log_config(cfg, cfg.log_params)
    if cfg.tag:
        logger.log_tag("tag", cfg.tag)
    logger.info("\n" + ConfigManager.pretty_text(cfg))
    with open(output_dir.parents[1] / "cmd_histry.log", "a") as f:
        print(get_cmd(), file=f)
    logger.log_artifact(output_dir.parents[1] / "cmd_histry.log")
    logger.log_artifact(output_dir / "config.yaml")

    do_test(cfg, output_dir, device, logger)

    messages = [
        f"host: {os.uname()[1]}",
        f"tag: {cfg.tag}",
        f"model: {cfg.model.name}",
        f"dataset: {cfg.test_dataset.name}",
        f"save: {str(output_dir)}",
        f"mlflow_url: {logger.get_mlflow_run_uri()}",
        f"wandb_url: {logger.get_wandb_run_uri()}",
    ]
    # Send Message to Slack
    post_slack(message="Finish Test\n" + "\n".join(messages))
    logger.info("Finish Test\n" + "\n".join(messages))
    logger.log_result_dir(output_dir, ignore_dirs=cfg.mlflow.ignore_artifact_dirs)
    logger.close()


if __name__ == "__main__":
    main()
