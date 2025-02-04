import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from src.config import ConfigManager, ExperimentConfig
from src.dataloaders import build_dataset
from src.evaluator import build_evaluator
from src.models import build_model
from src.optimizer import build_optimizer
from src.scheduler import build_lr_scheduler
from src.trainer import Trainer
from src.types import PathLike
from src.utils import (
    Logger,
    create_symlink,
    cuda_info,
    fix_seed,
    get_local_rank,
    get_local_size,
    get_world_rank,
    get_world_size,
    is_distributed,
    is_local_main_process,
    is_multi_node,
    is_world_main_process,
    load_model_weight,
    make_output_dirs,
    make_result_dirs,
    post_slack,
    save_lr_scheduler,
    save_model,
    save_model_info,
    save_optimizer,
    set_device,
)

from test import do_test  # isort: skip


def save_state(
    epoch: int,
    output_dir: PathLike,
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    iter_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
):
    # FSDPではmodel.state_dict()を呼び出した時に各プロセスの重みが集約されるので、
    # 全てのプロセスでsave_modelを呼び出す
    weight_path = f"{output_dir}/models/model_epoch_{epoch}.pth"
    save_model(model, weight_path)

    if is_world_main_process():
        # Create symlink to the latest model
        final_model_path = f"{output_dir}/models/model_final.pth"
        create_symlink(weight_path, final_model_path)

        optimizer_path = output_dir / "optimizers" / f"optimizer_epoch_{epoch}.pth"
        save_optimizer(optimizer, optimizer_path)
        cfg.optimizer.checkpoint = str(optimizer_path)

        # Create symlink to the latest optimizer
        final_optimizer_path = output_dir / "optimizers/optimizer_final.pth"
        create_symlink(optimizer_path, final_optimizer_path)

        if epoch_lr_scheduler:
            epoch_scheduler_path = output_dir / "schedulers" / f"epoch_scheduler_epoch_{epoch}.pth"
            save_lr_scheduler(epoch_lr_scheduler, epoch_scheduler_path)
            cfg.lr_scheduler.epoch_scheduler.checkpoint = str(epoch_scheduler_path)

            # Create symlink to the latest epoch scheduler
            final_epoch_scheduler_path = output_dir / "schedulers/epoch_scheduler_final.pth"
            create_symlink(epoch_scheduler_path, final_epoch_scheduler_path)

        if iter_lr_scheduler:
            iter_scheduler_path = output_dir / "schedulers" / f"iter_scheduler_epoch_{epoch}.pth"
            save_lr_scheduler(iter_lr_scheduler, iter_scheduler_path)
            cfg.lr_scheduler.iter_scheduler.checkpoint = str(iter_scheduler_path)

            # Create symlink to the latest iter scheduler
            final_iter_scheduler_path = output_dir / "schedulers/iter_scheduler_final.pth"
            create_symlink(iter_scheduler_path, final_iter_scheduler_path)

        cfg.last_epoch = epoch
        cfg.model.trained_weight = str(weight_path)
        ConfigManager.dump(cfg, output_dir / "config.yaml")


@record
def do_train(cfg: ExperimentConfig, device: torch.device, output_dir: Path, logger: Logger):
    fix_seed(cfg.seed + int(os.environ.get("LOCAL_RANK", -1)))

    # ###### Switch Epoch based or Iter based training #######
    if cfg.use_iter_loop:
        cfg.epoch = cfg.max_iter // cfg.step_iter
        if cfg.max_iter % cfg.step_iter != 0:
            cfg.epoch += 1
        cfg.save_interval = 1
        cfg.val_interval = 1
        logger.info(f"Iter based training ({cfg.epoch} iters)")
    else:
        logger.info(f"Epoch based training ({cfg.epoch} epochs)")

    # ###### Build Model #######
    logger.info("Building Model...")
    model = build_model(cfg, device, "train")
    # save initial model
    save_model(model, f"{output_dir}/models/model_init_0.pth")
    if is_world_main_process():
        model_summary = save_model_info(str(output_dir), model)
        logger.info(f"model architecture:\n{model}")
        logger.info(f"model summary:\n{model_summary}")
    logger.info("Complete Build Model")

    # ####### Build Dataset and Dataloader #######
    logger.info("Loading Dataset...")
    datasets, dataloaders, batched_transform = {}, {}, {}
    for phase in ["train", "val"]:
        datasets[phase], dataloaders[phase], batched_transform[phase] = build_dataset(
            cfg, phase=phase
        )
    logger.info("Complete Load Dataset")

    # ####### Build Optimizer #######
    logger.info("Building Optimizer...")
    optimizer = build_optimizer(cfg, model)
    logger.info("Complete Build Optimizer")

    # ####### Build LR Scheduler #######
    logger.info("Building LR Scheduler...")
    iter_lr_scheduler, epoch_lr_scheduler = build_lr_scheduler(cfg, optimizer)
    logger.info("Complete Build LR Scheduler")

    # ####### Build Evaluator #######
    logger.info("Building Evaluator...")
    evaluators = {}
    for phase in ["train", "val"]:
        evaluators[phase] = build_evaluator(cfg, phase)
        if evaluators[phase] is not None:
            evaluators[phase] = evaluators[phase].to(device)
    logger.info("Complete Build Evaluator")

    start_epoch = 0
    best_loss = 1e8
    best_epoch = 0
    # ####### Resume Training #######
    if cfg.last_epoch > 0:
        logger.info(f"Resume Training from Epoch {cfg.last_epoch}")
        start_epoch = cfg.last_epoch
        best_epoch = cfg.last_epoch
        load_model_weight(cfg.model.trained_weight, model)
        logger.info(f"Load model weight from {cfg.model.trained_weight}")

        # For ScheduleFree Optimizer
        if getattr(optimizer, "eval", None):
            optimizer.eval()
        optimizer.load_state_dict(
            torch.load(cfg.optimizer.checkpoint, map_location=device, weights_only=True)
        )
        logger.info(f"Load optimizer weight from {cfg.optimizer.checkpoint}")

        if epoch_lr_scheduler:
            epoch_lr_scheduler.load_state_dict(
                torch.load(
                    cfg.lr_scheduler.epoch_scheduler.checkpoint,
                    map_location=device,
                    weights_only=True,
                )
            )
            logger.info(
                f"Load epoch scheduler weight from {cfg.lr_scheduler.epoch_scheduler.checkpoint}"
            )
        if iter_lr_scheduler:
            iter_lr_scheduler.load_state_dict(
                torch.load(
                    cfg.lr_scheduler.iter_scheduler.checkpoint,
                    map_location=device,
                    weights_only=True,
                )
            )
            logger.info(
                f"Load iter scheduler weight from {cfg.lr_scheduler.iter_scheduler.checkpoint}"
            )
    # save initial model
    save_model(model, f"{output_dir}/models/model_init_{start_epoch}.pth")

    trainer = Trainer(
        epochs=cfg.epoch,
        device=device,
        datasets=datasets,
        dataloaders=dataloaders,
        batched_transforms=batched_transform,
        optimizer=optimizer,
        epoch_lr_scheduler=epoch_lr_scheduler,
        evaluators=evaluators,
        use_clip_grad=cfg.use_clip_grad,
        clip_grad=cfg.clip_grad_norm,
        gpu_multi_strategy=cfg.gpu.multi_strategy,
        iter_lr_scheduler=iter_lr_scheduler,
        use_amp=cfg.use_amp,
        amp_init_scale=cfg.amp_init_scale,
        amp_dtype=cfg.amp_dtype,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    logger.info("Start Training")
    for epoch in range(start_epoch, cfg.epoch):
        logger.phase = "train"
        logger.log_metric("Epoch", epoch + 1, epoch + 1)

        result = trainer.do_one_epoch(phase="train", current_epoch=epoch, model=model)
        logger.log_metrics(result.epoch_losses, epoch + 1, "train")
        logger.log_metric("Learning Rate", result.lr, epoch + 1, "train")
        logger.log_artifact(f"{output_dir}/train.log")
        if (epoch + 1) % cfg.save_interval == 0:
            save_state(
                epoch + 1, output_dir, cfg, model, optimizer, epoch_lr_scheduler, iter_lr_scheduler
            )

        if (epoch + 1) % cfg.val_interval == 0:
            result = trainer.do_one_epoch(phase="val", current_epoch=epoch, model=model)

            logger.log_metrics(result.epoch_losses, epoch + 1, "val")
            logger.log_metrics(result.metrics, epoch + 1, "val")
            logger.log_metric("Learning Rate", result.lr, epoch + 1, "val")
            weight_path = f"{output_dir}/models/model_epoch_{epoch+1}.pth"
            save_model(model, weight_path)
            if is_world_main_process() and result.epoch_losses["total_loss"] < best_loss:
                create_symlink(weight_path, output_dir / "models/model_best.pth")
                best_loss = result.epoch_losses["total_loss"]
                best_epoch = epoch + 1
                logger.info(f"Save model at best val loss({best_loss:.4f}) in Epoch {best_epoch}")

    # Finish Training Process
    save_state(epoch + 1, output_dir, cfg, model, optimizer, epoch_lr_scheduler, iter_lr_scheduler)
    if is_world_main_process():
        logger.log_artifact(output_dir / "models/model_final.pth")
        logger.log_artifact(output_dir / "optimizers/optimizer_final.pth")
        if epoch_lr_scheduler:
            logger.log_artifact(output_dir / "schedulers/epoch_scheduler_final.pth")
        if iter_lr_scheduler:
            logger.log_artifact(output_dir / "schedulers/iter_scheduler_final.pth")

        try:
            logger.log_history_figure()
        except Exception:
            logger.exception(f"Cannot draw graph. {traceback.format_exc()}")


@ConfigManager.argparse
def main(cfg: ExperimentConfig) -> None:
    # Set Local Rank for Multi GPU Training
    local_rank = get_local_rank()
    # set Device
    device = set_device(
        cfg.gpu.use,
        rank=local_rank,
        use_cudnn=cfg.gpu.use_cudnn,
        is_cpu=cfg.use_cpu,
        verbose=local_rank in [-1, 0],
        allow_tf32=cfg.gpu.use_tf32,
    )
    # DDP Mode
    if cfg.gpu.multi:
        dist.init_process_group(backend="nccl", init_method="env://")

    # create output dir
    prefix = f"{cfg.model.name}_{cfg.train_dataset.name}"
    prefix += "_" + cfg.tag if cfg.tag else ""
    output_dir = None
    if is_world_main_process():
        output_dir = make_output_dirs(
            Path(cfg.output) / cfg.train_dataset.name,
            prefix=prefix,
            child_dirs=["models", "optimizers", "schedulers"],
        )

    # Logging
    logger = Logger(
        output_dir,
        "train",
        level="INFO" if is_local_main_process() else "ERROR",
        use_mlflow=cfg.mlflow.use,
        use_wandb=cfg.wandb.use,
        mlflow_experiment_name=cfg.mlflow.experiment_name,
        wandb_project_name=cfg.wandb.project_name,
    )
    logger.log_config(cfg, cfg.log_params)
    if cfg.tag:
        logger.log_tag("tag", cfg.tag)
        logger.log_params({"tag": cfg.tag})
    if is_world_main_process():
        # Save config
        ConfigManager.dump(cfg, output_dir / "config.yaml")
    if is_distributed():
        logger.info("Use Distributed Data Parallel Training")
        if is_multi_node():
            logger.info("Multi Node Multi GPU Training")
            logger.info(
                f"hostname={os.uname()[1]}, "
                f"LOCAL_RANK={local_rank}, "
                f"LOCAL_WORLD_SIZE={get_local_size()}, "
                f"WORLD_RANK={get_world_rank()}, "
                f"WORLD_SIZE={get_world_size()}"
            )
        else:
            logger.info("Single Node Multi GPU Training")
            logger.info(
                f"hostname={os.uname()[1]}, "
                f"LOCAL_RANK={local_rank}, "
                f"LOCAL_WORLD_SIZE={get_local_size()}"
            )
    logger.info("\n" + ConfigManager.pretty_text(cfg))
    logger.log_artifacts(output_dir)
    cuda_info(logger=logger)

    try:
        do_train(cfg, device, output_dir, logger)
    except (Exception, KeyboardInterrupt) as e:
        if is_world_main_process():
            message = [
                f"host: {os.uname()[1]}",
                f"output: {output_dir}",
                f"last epoch: {cfg.last_epoch}",
                f"resume cmd: python train.py {output_dir / 'config.yaml'}",
                f"mlflow_url: {logger.get_mlflow_run_uri()}",
                f"train error: {e}\n{traceback.format_exc()}",
            ]
            logger.exception("\n".join(message))
            post_slack(channel="#error", message="\n".join(message))
            logger.log_result_dir(output_dir, ignore_dirs=cfg.mlflow.ignore_artifact_dirs)
            if is_distributed():
                dist.destroy_process_group()
        logger.close("FAILED")
        sys.exit(1)

    if not is_world_main_process():
        return
    # Clean Up multi gpu process
    if is_distributed():
        dist.destroy_process_group()
    torch.cuda.empty_cache()

    messages = [
        f"host: {os.uname()[1]}",
        f"tag: {cfg.tag}",
        f"model: {cfg.model.name}",
        f"train dataset: {cfg.train_dataset.name}",
        f"train output: {output_dir}",
        f"test cmd: python test.py {output_dir / 'config.yaml'}",
        f"mlflow_url: {logger.get_mlflow_run_uri()}",
        f"wandb_url: {logger.get_wandb_run_uri()}",
    ]
    post_slack(message="Finish Training:\n" + "\n".join(messages))
    logger.info("Finish Training:\n" + "\n".join(messages))
    cfg.model.trained_weight = str(output_dir / "models" / "model_best.pth")
    ConfigManager.dump(cfg, output_dir / "config.yaml")
    logger.log_artifact(cfg.model.trained_weight)
    logger.log_result_dir(output_dir, ignore_dirs=cfg.mlflow.ignore_artifact_dirs)

    # ####### Test #######
    output_result_dir = make_result_dirs(output_dir)
    logger.logger.add(output_result_dir / "test.log", level="INFO")
    logger.info("Start Test")
    logger.info(f"Output dir: {str(output_result_dir)}")
    logger.log_params(
        {"test_output": str(output_result_dir), "test_weight": cfg.model.trained_weight}
    )
    device = torch.device("cpu" if cfg.use_cpu else "cuda:0")
    try:
        do_test(cfg, output_result_dir, device, logger)
    except (Exception, KeyboardInterrupt) as e:
        if is_world_main_process():
            messages += [f"Test Error: {e}\n{traceback.format_exc()}\n"]
            logger.exception("\n".join(messages))
            post_slack(channel="#error", message="\n".join(messages))
        logger.close("FAILED")
        sys.exit(1)
    messages += [
        f"test output: {str(output_result_dir)}",
        f"test dataset: {cfg.test_dataset.name}",
    ]
    # Send Message to Slack
    post_slack(message="Finish Test\n" + "\n".join(messages))
    logger.info("Finish Test\n" + "\n".join(messages))
    ConfigManager.dump(cfg, output_result_dir / "config.yaml")
    logger.log_result_dir(output_dir, ignore_dirs=cfg.mlflow.ignore_artifact_dirs)
    logger.close()


if __name__ == "__main__":
    main()
