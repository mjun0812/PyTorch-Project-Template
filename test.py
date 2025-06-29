import json
import os
from pathlib import Path

import torch

from src.config import ConfigManager, ExperimentConfig
from src.dataloaders import build_dataloader, build_dataset, build_sampler
from src.evaluator import build_evaluator
from src.models import build_model
from src.runner import Tester
from src.transform import build_batched_transform, build_transforms
from src.utils import (
    JsonEncoder,
    Logger,
    get_cmd,
    load_model_weight,
    make_result_dirs,
    post_slack,
    setup_device,
)


def do_test(
    cfg: ExperimentConfig, output_dir: Path | str, device: torch.device, logger: Logger
) -> None:
    output_dir = Path(output_dir)

    logger.info("Loading Dataset...")
    transform = build_transforms(cfg.dataset.test.transforms)
    if cfg.dataset.test.batch_transforms is not None:
        batched_transform = build_batched_transform(cfg.dataset.test.batch_transforms).to(device)
    else:
        batched_transform = None
    dataset = build_dataset(cfg.dataset.test, transform)
    _, batch_sampler = build_sampler(dataset, phase="test", batch_size=cfg.batch)
    dataloader = build_dataloader(
        dataset,
        num_workers=cfg.num_worker,
        batch_sampler=batch_sampler,
        pin_memory=device.type != "mps",
    )
    logger.info("Complete Load Dataset")

    logger.info("Building Model...")
    model = build_model(cfg.model, "test")
    load_model_weight(cfg.model.checkpoint, model)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    logger.info("Complete Build model")

    logger.info("Building Evaluator...")
    evaluator = build_evaluator(cfg.evaluator.test)
    if evaluator is not None:
        evaluator = evaluator.to(device)
    logger.info("Complete Build Evaluator")

    if cfg.use_amp:
        logger.info(f"Using Mixed Precision with AMP (dtype: {cfg.amp_dtype})")

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
    logger.info(f"Speed(ms): {inference_speed * 1000:.5f}ms")
    logger.info(f"FPS: {1.0 / inference_speed:.2f}")
    metrics = results.metrics
    metrics.update({"Speed/ms": inference_speed * 1000, "fps": 1.0 / inference_speed})
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            metrics[name] = value.item()
    logger.log_metrics(metrics, None, "test")
    with open(output_dir / "result.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=JsonEncoder)

    tester.save_results(output_dir, results.targets, results.results)


def main(cfg: ExperimentConfig) -> None:
    # set Device
    device = setup_device(
        device_type=cfg.gpu.device, device_index=cfg.gpu.use, use_cudnn=cfg.gpu.use_cudnn
    )

    base_dir = Path(cfg.model.checkpoint).parents[1]
    output_dir = make_result_dirs(base_dir)

    ConfigManager.dump(cfg, output_dir / "config.yaml")
    logger = Logger(
        output_dir,
        "test",
        use_mlflow=cfg.log.use_mlflow,
        use_wandb=cfg.log.use_wandb,
        mlflow_experiment_name=cfg.log.mlflow_experiment_name,
        wandb_project_name=cfg.log.wandb_project_name,
        mlflow_run_name=f"{output_dir.parents[1].name}_{output_dir.name}",
    )
    logger.log_config(cfg, cfg.log.log_params)
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
        f"dataset: {cfg.dataset.test.name}",
        f"save: {output_dir!s}",
        f"mlflow_url: {logger.get_mlflow_run_uri()}",
        f"wandb_url: {logger.get_wandb_run_uri()}",
    ]
    # Send Message to Slack
    post_slack(message="Finish Test\n" + "\n".join(messages))
    logger.info("Finish Test\n" + "\n".join(messages))
    logger.log_result_dir(output_dir, ignore_dirs=cfg.log.mlflow_ignore_artifact_dirs)
    logger.close()


if __name__ == "__main__":
    cfg = ConfigManager.build()
    main(cfg)
