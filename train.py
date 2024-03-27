import glob
import os
import pprint
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from kunai.torch_utils import fix_seed, save_model, save_model_info, set_device
from natsort import natsorted
from torch.distributed.elastic.multiprocessing.errors import record

from src.dataloaders import build_dataset
from src.losses import build_loss
from src.models import build_model
from src.optimizer import build_optimizer
from src.scheduler import build_lr_scheduler
from src.trainer import Trainer
from src.utils import (
    Config,
    Logger,
    build_evaluator,
    error_handle,
    get_cmd,
    make_output_dirs,
    make_result_dirs,
    post_slack,
)

from test import do_test  # isort: skip


@record
def do_train(rank: int, cfg: dict, device: torch.device, output_dir: Path, logger: Logger):
    fix_seed(cfg.SEED + rank)

    # ###### Build Model #######
    model, model_ema = build_model(cfg, device, phase="train", rank=rank)
    if rank in [-1, 0]:
        save_model_path = output_dir / "models"
        # Model構造を出力
        save_model_info(str(output_dir), model)
        # save initial model
        save_model(model, save_model_path / "model_init_0.pth")
        logger.log_artifacts(output_dir)

    # ####### Build Dataset and Dataloader #######
    logger.info("Loading Dataset...")
    datasets, dataloaders, batched_transform = {}, {}, {}
    for phase in ["train", "val"]:
        datasets[phase], dataloaders[phase], batched_transform[phase] = build_dataset(
            cfg, phase=phase, rank=rank
        )
    logger.info("Complete Loading Dataset")

    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model, logger)
    if cfg.ITER_TRAIN:
        lr_scheduler = None
        _, iter_lr_scheduler = build_lr_scheduler(
            cfg.LR_SCHEDULER, optimizer, cfg.MAX_ITER, len(dataloaders["train"]), logger=logger
        )
        cfg.EPOCH = cfg.MAX_ITER // cfg.STEP_ITER
        cfg.SAVE_INTERVAL = 1
    else:
        iter_lr_scheduler, lr_scheduler = build_lr_scheduler(
            cfg.LR_SCHEDULER, optimizer, cfg.EPOCH, len(dataloaders["train"]), logger=logger
        )
    evaluator = build_evaluator(cfg, phase="train").to(device)
    if cfg.AMP:
        logger.info("Using Mixed Precision with AMP")

    trainer = Trainer(
        rank=rank,
        epochs=cfg.EPOCH,
        device=device,
        use_amp=cfg.AMP,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        use_clip_grad=cfg.USE_CLIP_GRAD,
        clip_grad=cfg.CLIP_GRAD_NORM,
        iter_lr_scheduler=iter_lr_scheduler,
        amp_init_scale=cfg.AMP_INIT_SCALE,
    )
    best_loss = 1e8
    best_epoch = 0

    logger.info("Start Training")
    for epoch in range(cfg.EPOCH):
        logger.info(f"Start Epoch {epoch+1}")
        logger.log_metric("Epoch", epoch + 1, epoch + 1)
        logger.phase = "train"

        if rank != -1:
            dataloaders[phase].sampler.set_epoch(epoch)

        result = trainer.do_one_epoch(
            phase="train",
            epoch=epoch,
            model=model,
            criterion=criterion,
            dataloader=dataloaders["train"],
            batched_transform=batched_transform["train"],
            model_ema=model_ema,
        )
        logger.info(trainer.build_epoch_log("train", epoch, **result))
        result["epoch_losses"]["Loss"] = result["epoch_losses"].pop("total_loss")
        logger.log_metrics(result["epoch_losses"], epoch + 1)
        logger.log_metric("Learning Rate", result["lr"], epoch + 1)
        logger.log_artifact(os.path.join(output_dir, "train.log"))
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0 and rank in [-1, 0]:
            # Save Model Weight
            save_model(model, save_model_path / f"model_epoch_{epoch+1}.pth")

        if (epoch + 1) % cfg.VAL_INTERVAL == 0:
            result = trainer.do_one_epoch(
                phase="val",
                epoch=epoch,
                model=model,
                criterion=criterion,
                dataloader=dataloaders["val"],
                batched_transform=batched_transform["val"],
                evaluator=evaluator,
            )
            logger.phase = "val"
            logger.info(trainer.build_epoch_log("val", epoch, **result))
            result["epoch_losses"]["Loss"] = result["epoch_losses"].pop("total_loss")
            logger.log_metrics(result["epoch_losses"], epoch + 1)
            logger.log_metrics(result["metrics"], epoch + 1)
            logger.log_metric("Learning Rate", result["lr"], epoch + 1)

            if result["epoch_losses"]["Loss"] < best_loss and rank in [-1, 0]:
                save_model(model, save_model_path / f"model_best_{epoch+1}.pth")
                best_loss = result["epoch_losses"]["Loss"]
                best_epoch = epoch + 1
                logger.info(f"Save model at best val loss({best_loss:.4f}) in Epoch {best_epoch}")

    # Finish Training Process below
    if rank in [-1, 0]:
        try:
            logger.log_history_figure()
        except Exception:
            logger.error(f"Cannot draw graph. {traceback.format_exc()}")
        save_model(model, save_model_path / f"model_final_{cfg.EPOCH}.pth")
        logger.log_artifact(save_model_path / f"model_final_{cfg.EPOCH}.pth")
    return best_loss


@Config.main
def main(cfg):
    # Set Local Rank for Multi GPU Training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # set Device
    device = set_device(
        cfg.GPU.USE,
        rank=local_rank,
        use_cudnn=cfg.CUDNN,
        is_cpu=cfg.CPU,
        verbose=local_rank in [0, -1],
    )

    if local_rank in [0, -1]:
        # make Output dir
        prefix = f"{cfg.MODEL.NAME}_{cfg.TRAIN_DATASET.NAME}"
        prefix += "_" + cfg.TAG if cfg.TAG else ""
        output_dir = make_output_dirs(
            os.path.join(cfg.OUTPUT, cfg.TRAIN_DATASET.NAME),
            prefix=prefix,
            child_dirs=["figs", "models"],
        )
        output_dir = Path(output_dir)

        # Logging
        logger = Logger(str(output_dir), str(output_dir / "train.log"), "train", "INFO")
        if cfg.USE_MLFLOW:
            logger.setup_mlflow(f"{output_dir.name}_{prefix}", cfg.MLFLOW_EXPERIMENT_NAME)

        # Save config
        Config.dump(cfg, output_dir / "config.yaml")
        # Execute CLI command
        with open(output_dir / "cmd_histry.log", "a") as f:
            print(get_cmd(), file=f)
    else:
        output_dir = None
        logger = Logger(None, None, "train", "ERROR")

    logger.info("\n" + Config.pretty_text(cfg))
    logger.log_artifacts(output_dir)

    # DDP Mode
    if bool(cfg.GPU.MULTI):
        dist.init_process_group(backend="nccl", init_method="env://")
        logger.info("Use Distributed Data Parallel Training")
        logger.info(
            f"hostname={os.uname()[1]}, LOCAL_RANK={local_rank}, "
            f"RANK={dist.get_rank()}, WORLD_SIZE={dist.get_world_size()}"
        )

    try:
        result = do_train(local_rank, cfg, device, output_dir, logger)
    except (Exception, KeyboardInterrupt) as e:
        if local_rank in [0, -1]:
            error_handle(e, "Train", f"Output: {str(output_dir)}")
        logger.log_result_dir(output_dir, ignore_dirs=cfg.MLFLOW_IGNORE_DIRS)
        logger.close("FAILED")
        sys.exit(1)

    # Clean Up multi gpu process
    if local_rank != -1:
        dist.destroy_process_group()
    torch.cuda.empty_cache()

    if local_rank in [0, -1]:
        message_dict = {
            "host": os.uname()[1],
            "tag": cfg.TAG,
            "model": cfg.MODEL.NAME,
            "dataset": cfg.TRAIN_DATASET.NAME,
            "Train save": str(output_dir),
            "Val Loss": f"{result:7.3f}",
            "Test Cmd": f"python test.py {str(output_dir / 'config.yaml')}",
        }
        message = pprint.pformat(message_dict, width=150)
        # Send Message to Slack
        post_slack(message=f"Finish Training\n{message}")
        logger.info(f"Finish Training {message}")

        # Prepare config for Test
        cfg.MODEL.WEIGHT = natsorted(glob.glob(str(output_dir / "models" / "model_best_*.pth")))[
            -1
        ]
        cfg.GPU.MULTI = False
        cfg.GPU.USE = 0
        Config.dump(cfg, output_dir / "config.yaml")
        logger.log_artifact(cfg.MODEL.WEIGHT)
        logger.log_result_dir(str(output_dir), ignore_dirs=cfg.MLFLOW_IGNORE_DIRS)

        # Test
        logger.info("Start Test")
        logger.phase = "test"
        logger.log_tag("model_weight_test", cfg.MODEL.WEIGHT)
        output_result_dir = Path(make_result_dirs(cfg.MODEL.WEIGHT))
        device = torch.device("cpu" if cfg.CPU else "cuda:0")
        try:
            result = do_test(cfg, output_result_dir, device, logger)
        except (Exception, KeyboardInterrupt) as e:
            error_handle(e, "Test", f"Output: {output_dir}")
            logger.close("FAILED")
            sys.exit(1)
        message_dict.update({"Test save": str(output_result_dir), "result": result})
        message = pprint.pformat(message_dict, width=150)
        # Send Message to Slack
        post_slack(message=f"Finish Test\n{message}")
        logger.info(f"Finish Test {message}")
        Config.dump(cfg, os.path.join(output_result_dir, "config.yaml"))
        logger.log_result_dir(str(output_result_dir), ignore_dirs=cfg.MLFLOW_IGNORE_DIRS)
        logger.close()
    return result


if __name__ == "__main__":
    main()
