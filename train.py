import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from src.config import ConfigManager, DatasetConfig, ExperimentConfig
from src.dataloaders import build_dataloader, build_dataset, build_sampler
from src.evaluator import build_evaluator
from src.models import (
    BaseModel,
    build_model,
    log_model_parameters,
    setup_ddp_model,
    setup_fsdp_model,
)
from src.optimizer import build_optimizer
from src.scheduler import build_lr_scheduler
from src.trainer import Trainer, TrainerParams
from src.transform import (
    BatchedTransformCompose,
    build_batched_transform,
    build_transforms,
)
from src.utils import (
    Logger,
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
    make_output_dirs,
    make_result_dirs,
    post_slack,
    save_model_info,
    set_device,
)

from test import do_test  # isort: skip


def build_train_model(cfg: ExperimentConfig, device: torch.device, logger: Logger) -> BaseModel:
    model = build_model(cfg.model, phase="train")
    model = model.to(device)
    log_model_parameters(model)

    if is_distributed():
        if cfg.gpu.multi_strategy == "ddp":
            logger.info("Use DistributedDataParallel Training")
            if cfg.model.use_sync_bn:
                logger.info("Use Sync BatchNorm")
            model = setup_ddp_model(cfg, model)
        elif cfg.gpu.multi_strategy == "fsdp":
            logger.info("Use FullyShardedDataParallel Training")
            model = setup_fsdp_model(cfg, model)
    elif cfg.gpu.multi and cfg.gpu.multi_strategy == "dp" and torch.cuda.device_count() > 1:
        logger.info("Use DataParallel Training")
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    if cfg.use_compile and torch.__version__ >= "2.0.0":
        logger.info("Use Torch Compile")
        model = torch.compile(model, backend=cfg.compile_backend)

    return model


def build_phase_dataloader(
    cfg: ExperimentConfig, phase: str, logger: Logger
) -> tuple[Dataset, DataLoader, Optional[BatchedTransformCompose], Sampler]:
    cfg_dataset: DatasetConfig = cfg.dataset.get(phase)

    transform = build_transforms(cfg_dataset.transforms)
    batched_transform = None
    if cfg_dataset.batch_transforms is not None:
        batched_transform = build_batched_transform(cfg_dataset.batch_transforms)

    dataset = build_dataset(cfg_dataset, transform, cfg.use_ram_cache, cfg.ram_cache_size_gb)
    sampler, batch_sampler = build_sampler(dataset, phase, cfg.batch, cfg.dataset.batch_sampler)
    dataloader = build_dataloader(
        dataset,
        cfg.num_worker,
        batch_sampler,
        use_iter_loop=cfg.use_iter_loop and phase == "train",
        max_iter=cfg.max_iter if phase == "train" else None,
        step_iter=cfg.step_iter if phase == "train" else None,
    )

    phase_cap = phase.capitalize()
    logger.info(f"{phase_cap} {cfg_dataset.name} Dataset sample num: {len(dataset)}")
    logger.info(f"{phase_cap} transform: {transform}")
    logger.info(f"{phase_cap} batched transform: {batched_transform}")
    return dataset, dataloader, batched_transform, sampler


@record
def do_train(cfg: ExperimentConfig, device: torch.device, output_dir: Path, logger: Logger):
    fix_seed(cfg.seed + int(os.environ.get("LOCAL_RANK", -1)))

    # ###### Build Model #######
    model = build_train_model(cfg, device, logger)
    if is_world_main_process():
        model_summary = save_model_info(str(output_dir), model)
        logger.info(f"model architecture:\n{model}")
        logger.info(f"model summary:\n{model_summary}")

    # ####### Build Dataset, Dataloader, Batched Transform, Sampler #######
    datasets, dataloaders, batched_transforms, samplers = {}, {}, {}, {}
    for phase in ["train", "val"]:
        datasets[phase], dataloaders[phase], batched_transforms[phase], samplers[phase] = (
            build_phase_dataloader(cfg, phase, logger)
        )

    # ####### Build Optimizer #######
    optimizer = build_optimizer(cfg.optimizer, model)
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Optimizer Group: {cfg.optimizer.group}")
    logger.info(f"Optimizer Learning Rate: {cfg.optimizer.lr}")

    # ####### Build LR Scheduler #######
    iter_lr_scheduler, epoch_lr_scheduler = build_lr_scheduler(
        cfg.lr_scheduler, optimizer, cfg.epoch, cfg.max_iter
    )
    if epoch_lr_scheduler:
        logger.info(f"Epoch Scheduler: {cfg.lr_scheduler.epoch_scheduler.class_name}")
    if iter_lr_scheduler:
        logger.info(f"Iter Scheduler: {cfg.lr_scheduler.iter_scheduler.class_name}")

    # ####### Build Evaluator #######
    evaluators = {}
    for phase in ["train", "val"]:
        if cfg.evaluator.get(phase) is not None:
            evaluators[phase] = build_evaluator(cfg.evaluator.get(phase)).to(device)
            logger.info(f"{phase.capitalize()} Evaluators: {evaluators[phase]}")
        else:
            evaluators[phase] = None

    # ###### Setup Training Params #######
    if cfg.use_iter_loop:
        cfg.epoch = cfg.max_iter // cfg.step_iter
        if cfg.max_iter % cfg.step_iter != 0:
            cfg.epoch += 1
        cfg.save_interval = 1
        cfg.val_interval = 1
        logger.info(f"Iter based training ({cfg.epoch} iters)")
    else:
        logger.info(f"Epoch based training ({cfg.epoch} epochs)")
    trainer_params = TrainerParams(
        output_dir=output_dir,
        epochs=cfg.epoch,
        use_clip_grad=cfg.use_clip_grad,
        clip_grad=cfg.clip_grad_norm,
        gpu_multi_strategy=cfg.gpu.multi_strategy,
        use_amp=cfg.use_amp,
        amp_init_scale=cfg.amp_init_scale,
        amp_dtype=cfg.amp_dtype,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        greater_is_better=cfg.greater_is_better,
        metric_for_best_model=cfg.metric_for_best_model,
        save_interval=cfg.save_interval,
        val_interval=cfg.val_interval,
    )
    if trainer_params.use_amp:
        logger.info(f"Using Mixed Precision with AMP (dtype: {trainer_params.amp_dtype})")
    if trainer_params.gradient_accumulation_steps > 1:
        logger.info(f"Gradient Accumulation Steps: {trainer_params.gradient_accumulation_steps}")
    for phase in ["train", "val"]:
        total_iters = len(dataloaders[phase]) * trainer_params.epochs
        logger.info(f"Total {phase} Iterations per GPU: {total_iters}")
        logger.info(f"Total {phase} Iterations per World: {total_iters * get_world_size()}")

    trainer = Trainer(
        cfg=cfg,
        params=trainer_params,
        device=device,
        logger=logger,
        model=model,
        datasets=datasets,
        dataloaders=dataloaders,
        samplers=samplers,
        batched_transforms=batched_transforms,
        optimizer=optimizer,
        epoch_lr_scheduler=epoch_lr_scheduler,
        iter_lr_scheduler=iter_lr_scheduler,
        evaluators=evaluators,
    )

    start_epoch = 0

    # ####### Resume Training #######
    if cfg.last_epoch > 0:
        logger.info(f"Resume Training from Epoch {cfg.last_epoch}")
        logger.info(f"best metric: {cfg.best_metric}")
        logger.info(f"best epoch: {cfg.best_epoch}")
        start_epoch = cfg.last_epoch

        if cfg.lr_scheduler.epoch_scheduler:
            epoch_scheduler_checkpoint = cfg.lr_scheduler.epoch_scheduler.checkpoint
        else:
            epoch_scheduler_checkpoint = None
        if cfg.lr_scheduler.iter_scheduler:
            iter_scheduler_checkpoint = cfg.lr_scheduler.iter_scheduler.checkpoint
        else:
            iter_scheduler_checkpoint = None
        trainer.load_state(
            start_epoch,
            cfg.best_metric,
            cfg.best_epoch,
            cfg.model.checkpoint,
            cfg.optimizer.checkpoint,
            epoch_scheduler_checkpoint,
            iter_scheduler_checkpoint,
        )

    # save initial state
    trainer.save_state(start_epoch, output_dir)

    logger.info("Start Training")
    trainer.train(start_epoch)


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
    prefix = cfg.model.name
    if cfg.tag:
        prefix += f"_{cfg.tag}"
    output_dir = None
    if is_world_main_process():
        output_dir = make_output_dirs(
            Path(cfg.output) / cfg.dataset.train.name,
            prefix=prefix,
            child_dirs=["models", "optimizers", "schedulers"],
        )

    # create logger
    logger = Logger(
        output_dir,
        "train",
        level="INFO" if is_local_main_process() else "ERROR",
        use_mlflow=cfg.log.use_mlflow and is_world_main_process(),
        use_wandb=cfg.log.use_wandb and is_world_main_process(),
        mlflow_experiment_name=cfg.log.mlflow_experiment_name,
        wandb_project_name=cfg.log.wandb_project_name,
    )
    if cfg.tag:
        logger.log_tag("tag", cfg.tag)
        logger.log_params({"tag": cfg.tag})
    # logging Device Info
    cuda_info(logger=logger)
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

    # save config
    if is_world_main_process():
        ConfigManager.dump(cfg, output_dir / "config.yaml")
    logger.info("\n" + ConfigManager.pretty_text(cfg))
    logger.log_config(cfg, cfg.log.log_params)
    logger.log_artifacts(output_dir)

    messages = [
        f"host: {os.uname()[1]}",
        f"output: {output_dir}",
        f"tag: {cfg.tag}",
        f"model: {cfg.model.name}",
        f"train dataset: {cfg.dataset.train.name}",
        f"resume cmd: python train.py {output_dir}/config.yaml",
        f"mlflow_url: {logger.get_mlflow_run_uri()}",
        f"wandb_url: {logger.get_wandb_run_uri()}",
    ]

    try:
        do_train(cfg, device, output_dir, logger)
    except (Exception, KeyboardInterrupt) as e:
        if is_world_main_process():
            messages += [f"train error: {e}\n{traceback.format_exc()}"]
            logger.exception("\n".join(messages))
            post_slack(channel="#error", message="\n".join(messages))
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

    messages += [f"test cmd: python test.py {output_dir}/config.yaml"]
    post_slack(message="Finish Training:\n" + "\n".join(messages))
    logger.info("Finish Training:\n" + "\n".join(messages))
    logger.log_result_dir(output_dir, ignore_dirs=cfg.log.mlflow_ignore_artifact_dirs)

    # ####### Test #######
    output_result_dir = make_result_dirs(output_dir)
    logger.logger.add(output_result_dir / "test.log", level="INFO")
    logger.info("Start Test")
    logger.info(f"Output dir: {str(output_result_dir)}")
    logger.log_params({"test_output": str(output_result_dir), "test_weight": cfg.model.checkpoint})
    try:
        do_test(cfg, output_result_dir, device, logger)
    except (Exception, KeyboardInterrupt) as e:
        messages += [f"Test Error: {e}\n{traceback.format_exc()}\n"]
        logger.exception("\n".join(messages))
        post_slack(channel="#error", message="\n".join(messages))
        logger.close("FAILED")
        sys.exit(1)
    messages += [
        f"test output: {str(output_result_dir)}",
        f"test dataset: {cfg.dataset.test.name}",
    ]
    # Send Message to Slack
    post_slack(message="Finish Test\n" + "\n".join(messages))
    logger.info("Finish Test\n" + "\n".join(messages))
    logger.log_result_dir(output_dir, ignore_dirs=cfg.log.mlflow_ignore_artifact_dirs)
    logger.close()


if __name__ == "__main__":
    cfg = ConfigManager.build()
    main(cfg)
