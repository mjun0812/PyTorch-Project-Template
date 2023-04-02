import glob
import logging
import os
import sys
import traceback
from pathlib import Path
import pprint
import datetime
import shutil

import torch
import torch.distributed as dist

from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from natsort import natsorted

from kunai.hydra_utils import set_hydra
from kunai.torch_utils import (
    fix_seed,
    save_model,
    save_model_info,
    set_device,
    check_model_parallel,
)
from kunai.utils import get_cmd, get_git_hash, setup_logger, make_output_dirs

from src.dataloaders import build_dataset
from src.losses import build_loss
from src.models import build_model
from src.utils import (
    TrainLogger,
    post_slack,
    make_result_dirs,
    reduce_tensor,
)
from src.optimizer import build_optimizer
from src.scheduler import build_lr_scheduler
from test import do_test


# Get root logger
logger = logging.getLogger()


def load_last_weight(cfg, model):
    """Load PreTrained or Continued Model

    Args:
        model (torch.nn.Model): Load model
        weight (str): PreTrained weight path
    """
    if cfg.CONTINUE_TRAIN:
        weight_path = cfg.MODEL.WEIGHT
    elif cfg.MODEL.PRE_TRAINED and cfg.MODEL.PRE_TRAINED_WEIGHT:
        # Train from Pretrained weight
        weight_path = cfg.MODEL.PRE_TRAINED_WEIGHT
    else:
        return

    if check_model_parallel(model):
        model = model.module
    device = next(model.parameters()).device

    check_point = torch.load(weight_path, map_location=device)
    # for torch.compile model
    state_dict = {}
    for k, v in check_point.items():
        state_dict[k.replace("_orig_mod.", "").replace("module.", "")] = v
    missing, unexpexted = model.load_state_dict(state_dict, strict=False)

    logger.info(f"Load weight from {weight_path}")
    logger.info(f"Missing model key: {missing}")
    logger.info(f"Unexpected model key: {unexpexted}")


def do_train(rank, cfg, device, output_dir, writer):
    """Training Script
       このFunctionでSingle GPUとMulti GPUの両方に対応しています．

    Args:
        rank (int): Processランク．Single GPUのときは-1，Multi GPUのときは0,1...
                    この値でMultiとSingleの区別，Master Prosessの区別を行っている．
                    MultiGPUの場合，MasterProcess以外でFile IOを行うのはNGなので
                    この値でFileIOするかを判定する。
                    rank = [-1,0]のときにIOをして、それ以外ではIOは行わない。
                    IOはファイル書き込み以外に標準出力も含む
        cfg (OmegaConf): Hydra Conf
        output_dict (dict): resultを格納するDirectry Path
    """

    fix_seed(100 + rank)
    save_model_path = Path(output_dir, "models")

    # ###### Build Model #######
    model, model_ema = build_model(cfg, device, phase="train", rank=rank)
    load_last_weight(cfg, model)
    if cfg.COMPILE and torch.__version__ >= "2.0.0":
        import torch._dynamo as dynamo  # noqa

        logger.info("Use Torch Dynamo Compile")
        dynamo.reset()
        model = torch.compile(model, backend=cfg.COMPILE_BACKEND)

    # ####### Build Dataset and Dataloader #######
    logger.info("Loading Dataset...")
    datasets, dataloaders = {}, {}
    for phase in ["train", "val"]:
        datasets[phase], dataloaders[phase] = build_dataset(cfg, phase=phase, rank=rank)
    logger.info("Complete Loading Dataset")

    # ###### Logging ######
    if rank in [-1, 0]:
        # Model構造を出力
        save_model_info(output_dir, model)
        # save initial model
        save_model(model, save_model_path / "model_init_0.pth")

    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.AMP)
    if cfg.AMP:
        logger.info("Using Mixed Precision with AMP")

    max_epoch = cfg.EPOCH
    best_loss = 1e8
    best_epoch = 0

    logger.info("Start Training")
    for epoch in range(max_epoch):
        logger.info(f"Start Epoch {epoch+1}")
        for phase in ["train", "val"]:
            hist_epoch_loss = torch.tensor(0)

            # Skip Validation
            if phase == "val" and ((epoch + 1) % cfg.VAL_INTERVAL != 0):
                continue

            model.train(phase == "train")
            if phase == "train" and rank != -1:
                dataloaders[phase].sampler.set_epoch(epoch)
            if phase == "train" and rank in [-1, 0]:
                writer.log_metric("Epoch", epoch + 1, phase, epoch + 1)

            # Set progress bar
            progress_bar = enumerate(dataloaders[phase])
            if rank in [-1, 0]:
                progress_bar = tqdm(
                    progress_bar, total=len(dataloaders[phase]), dynamic_ncols=True
                )

            for _, data in progress_bar:
                with torch.set_grad_enabled(phase == "train"):
                    data = data.to(device, non_blocking=True).float()

                    with torch.autocast(
                        device_type="cuda" if not cfg.CPU else "cpu",
                        enabled=cfg.AMP,
                        dtype=torch.float16,
                    ):
                        y = model(data)
                        loss = criterion(y, data)

                    if phase == "train":
                        scaler.scale(loss["Loss"]).backward()
                        if cfg.USE_CLIP_GRAD:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD_NORM)
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad()
                    if cfg.MODEL_EMA:
                        model_ema.update(model)

                    for key in loss.keys():
                        if rank != -1:
                            loss[key] = reduce_tensor(loss[key]) / dist.get_world_size()
                        hist_epoch_loss[key] = (
                            hist_epoch_loss.get(key, 0.0) + loss[key].item() * data.shape[0]
                        )
                    if rank in [-1, 0]:
                        description = f"Epoch: {epoch + 1:3}/{max_epoch:3}."
                        for k, v in loss.items():
                            description += f" {k.capitalize()}: {v.item():8.4f}"
                        progress_bar.set_description(description)

            # Finish Train or Val Epoch Process below
            for k, v in hist_epoch_loss.items():
                hist_epoch_loss[k] = hist_epoch_loss[k] / (len(dataloaders[phase]) * cfg.BATCH)
            epoch_loss = hist_epoch_loss["Loss"]
            lr = optimizer.param_groups[0]["lr"]

            if phase == "train":
                if cfg.LR_SCHEDULER.NAME == "ReduceLROnPlateau":
                    scheduler.step(epoch_loss)
                elif cfg.LR_SCHEDULER.NAME == "CosineLRScheduler":
                    scheduler.step(epoch + 1)
                else:
                    scheduler.step()

            if rank in [-1, 0]:
                description = f"{phase.capitalize()} Epoch: {(epoch + 1):3}/{max_epoch:3}. "
                for k, v in hist_epoch_loss.items():
                    description += f" {k}: {v:8.4f}"
                description += (
                    f" GPU: {torch.cuda.memory_reserved(device) / 1e9:.1f}GB LR: {lr:.4e}"
                )
                logger.info(description)
                hist_epoch_loss.update({"Loss": epoch_loss, "Learning Rate": lr})
                writer.log_metrics(
                    phase, list(hist_epoch_loss.keys()), list(hist_epoch_loss.values()), epoch + 1
                )

                if phase == "train" and ((epoch + 1) % cfg.SAVE_INTERVAL == 0):
                    # Save Model Weight
                    save_model(model, save_model_path / f"model_epoch_{epoch+1}.pth")
                elif phase == "val":
                    # Save best val Loss Model
                    if epoch_loss < best_loss:
                        shutil.rmtree(
                            save_model_path / f"model_best_{best_epoch}.pth", ignore_errors=True
                        )
                        save_model(model, save_model_path / f"model_best_{epoch+1}.pth")
                        best_loss = epoch_loss
                        best_epoch = epoch + 1
                        logger.info(
                            f"Save model at best val loss({best_loss:.4f}) in Epoch {best_epoch}"
                        )
                    # early stopping (check val_loss)
                    if epoch + 1 - best_epoch > int(cfg.EARLY_STOP_PATIENCE) > 0:
                        logger.info(
                            f"Stop training at epoch {epoch + 1}. The lowest loss achieved is {best_loss}"
                        )
                        break

    # Finish Training Process below
    if rank in [-1, 0]:
        try:
            writer.log_history_figure()
        except Exception:
            logger.error(f"Cannot draw graph. {traceback.format_exc()}")
        save_model(model, os.path.join(save_model_path, f"model_final_{max_epoch}.pth"))
        writer.log_artifact(os.path.join(save_model_path, f"model_final_{max_epoch}.pth"))
    return best_loss


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    # Set Local Rank for Multi GPU Training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Hydra Setting
    set_hydra(cfg, verbose=local_rank in [0, -1])
    # set Device
    device = set_device(
        cfg.GPU.USE, rank=local_rank, is_cpu=cfg.CPU, verbose=local_rank in [0, -1]
    )

    if local_rank in [0, -1]:
        # make Output dir
        prefix = f"{cfg.MODEL.NAME}_{cfg.DATASET.NAME}"
        if cfg.TAG:
            prefix += f"_{cfg.TAG}"
        output_dir = make_output_dirs(
            os.path.join(cfg.OUTPUT, cfg.DATASET.NAME),
            prefix=prefix,
            child_dirs=["figs", "models"],
        )

        # Logging
        setup_logger(os.path.join(output_dir, "train.log"))
        logger.info(f"Command: {get_cmd()}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Git Hash: {get_git_hash()}")

        # Hydra config
        OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

        # Execute CLI command
        with open(os.path.join(output_dir, "cmd_histry.log"), "a") as f:
            print(get_cmd(), file=f)

        # Set Tensorboard, MLflow
        writer = TrainLogger(cfg, output_dir, os.path.dirname(output_dir))
        writer.log_artifact(os.path.join(output_dir, "config.yaml"))
        writer.log_artifact(os.path.join(output_dir, "cmd_histry.log"))
    else:
        output_dir = ""
        setup_logger()
        writer = None

    # PyTorch A6000 Bug Fix: GPU間通信をP2PからPCI or NVLINKに変更する
    # os.environ["NCCL_P2P_DISABLE"] = "1"

    # DDP Mode
    if bool(cfg.GPU.MULTI):
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=2700)
        )
        logging.info("Use Distributed Data Parallel Training")
        logging.info(
            f"hostname={os.uname()[1]}, LOCAL_RANK={local_rank}, "
            f"RANK={dist.get_rank()}, WORLD_SIZE={dist.get_world_size()}"
        )

    try:
        result = do_train(local_rank, cfg, device, output_dir, writer)
    except (Exception, KeyboardInterrupt) as e:
        if local_rank in [0, -1]:
            logger.error(f"{e}\n{traceback.format_exc()}")
            post_slack(
                channel="#error",
                message=f"Error Train\n{e}\n{traceback.format_exc()}\nOutput: {output_dir}",
            )
            writer.log_result_dir(output_dir)
            writer.close("FAILED")
        if local_rank != -1:
            dist.destroy_process_group()
        sys.exit(1)

    if local_rank in [0, -1]:
        message_dict = {
            "host": os.uname()[1],
            "tag": cfg.TAG,
            "model": cfg.MODEL.NAME,
            "dataset": cfg.DATASET.NAME,
            "Train save": output_dir,
            "Val Loss": f"{result:7.3f}",
            "Test Cmd": f"python test.py -cp {output_dir}",
        }
        message = pprint.pformat(message_dict, width=150)
        # Send Message to Slack
        post_slack(message=f"Finish Training\n{message}")
        logger.info(f"Finish Training {message}")

        # Prepare config for Test
        cfg.MODEL.WEIGHT = natsorted(
            glob.glob(os.path.join(output_dir, "models", "model_best_*.pth"))
        )[-1]
        writer.log_artifact(cfg.MODEL.WEIGHT)
        if local_rank == 0:
            cfg.GPU.MULTI = False
            cfg.GPU.USE = 0
        OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))
        writer.log_artifact(os.path.join(output_dir, "config.yaml"))
        writer.log_result_dir(output_dir)

    # Clean Up multi gpu process
    if local_rank != -1:
        dist.destroy_process_group()
    torch.cuda.empty_cache()

    # Test
    if local_rank in [0, -1]:
        logger.info("Start Test")
        writer.log_tag("model_weight_test", cfg.MODEL.WEIGHT)
        output_result_dir = make_result_dirs(cfg.MODEL.WEIGHT)
        if cfg.CPU:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
        try:
            result = do_test(cfg, output_result_dir, device, writer)
        except (Exception, KeyboardInterrupt) as e:
            logger.error(f"{e}\n{traceback.format_exc()}")
            post_slack(
                channel="#error",
                message=f"Error Test\n{e}\n{traceback.format_exc()}\nOutput: {output_dir}",
            )
            writer.close("FAILED")
            sys.exit(1)
        message_dict["Test save"] = output_result_dir
        message_dict["result"] = result
        message = pprint.pformat(message_dict, width=150)
        # Send Message to Slack
        post_slack(message=f"Finish Test\n{message}")
        logger.info(f"Finish Test {message}")
        OmegaConf.save(cfg, os.path.join(output_result_dir, "config.yaml"))
        writer.log_result_dir(output_dir)
        writer.log_result_dir(output_result_dir)
        writer.close()
    return result


if __name__ == "__main__":
    main()
