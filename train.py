import glob
import logging
import os
import sys
import traceback
import pathlib
import pprint

import torch
import torch.distributed as dist

from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from natsort import natsorted
from clearml import Task
from timm.utils import ModelEmaV2

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
    build_lr_scheduler,
    build_optimizer,
    post_slack,
    make_result_dirs,
)
from test import do_test


# Get root logger
logger = logging.getLogger()


def load_last_weight(cfg, model):
    """Load PreTrained or Continued Model

    Args:
        model (torch.nn.Model): Load model
        weight (str): PreTrained weight path

    Returns:
        model: Loaded Model
        last_epoch: number of last epoch from weight file name
                    ex. 'weight/model_epoch_15.pth' return '15'
    """
    if cfg.CONTINUE_TRAIN:
        weight_path = cfg.MODEL.WEIGHT
    elif cfg.MODEL.PRE_TRAINED and cfg.MODEL.PRE_TRAINED_WEIGHT:
        # Train from Pretrained weight
        weight_path = cfg.MODEL.PRE_TRAINED_WEIGHT
    else:
        return 0

    try:
        # If continue train, get final Epoch number
        last_epoch = os.path.basename(weight_path).split("_")[-1]
        last_epoch = int(last_epoch)
    except Exception:
        last_epoch = 0

    device = next(model.parameters()).device
    try:
        if check_model_parallel(model):
            model = model.module
        model.load_state_dict(torch.load(weight_path, map_location=device))
    except RuntimeError:
        # fine tuning
        logger.warning("Class num changed from loading weights. Do FineTuning?")

    logger.info(f"Load weight from {weight_path}")
    return last_epoch


def do_train(rank, cfg, output_dir, writer):
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

    initial_seed = fix_seed(100 + rank)
    save_model_path = pathlib.Path(output_dir, "models")

    # Set Device
    if cfg.CPU:
        device = torch.device("cpu")
    elif rank != -1:
        device = torch.device(rank)
        torch.cuda.set_device(rank)
    else:
        device = torch.device(0)
        torch.cuda.set_device(0)

    # ###### Build Model #######
    model = build_model(cfg, device, rank=rank)
    last_epoch = load_last_weight(cfg, model)  # Train from exist weight
    if cfg.MODEL_EMA:
        model_ema = ModelEmaV2(model, decay=0.9998)

    # ####### Build Dataset and Dataloader #######
    logger.info("Loading Dataset...")
    datasets = {}
    dataloaders = {}
    for phase in ["train", "val"]:
        datasets[phase], dataloaders[phase] = build_dataset(cfg, phase=phase, rank=rank)
    logger.info("Complete Loading Dataset")

    # ###### Logging ######
    metrics = ["Loss", "Learning Rate"]
    if rank in [-1, 0]:
        # Model構造を出力
        save_model_info(output_dir, model)
        # save initial model
        save_model(model, save_model_path / f"model_init_{last_epoch}.pth")

    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model, torch.cuda.device_count())
    scheduler = build_lr_scheduler(cfg, optimizer)
    if cfg.AMP:
        logger.info("Using Mixed Precision with AMP")
        scaler = torch.cuda.amp.GradScaler()

    max_epoch = cfg.EPOCH + last_epoch
    best_loss = 1e8
    best_epoch = 0

    logger.info("Start Training")
    for epoch in range(last_epoch, max_epoch, 1):
        logger.info(f"Start Epoch {epoch+1}")
        np.random.seed(initial_seed + epoch + rank)
        for phase in ["train", "val"]:
            hist_epoch_loss = 0

            # Skip Validation
            if phase == "val" and ((epoch + 1) % cfg.VAL_INTERVAL != 0):
                continue

            model.train(phase == "train")
            if phase == "train" and rank != -1:
                dataloaders[phase].sampler.set_epoch(epoch)

            # Set progress bar
            progress_bar = enumerate(dataloaders[phase])
            if rank in [-1, 0]:
                progress_bar = tqdm(progress_bar, total=len(dataloaders[phase]))

            for _, data in progress_bar:
                with torch.set_grad_enabled(phase == "train"):
                    data = data.to(device, non_blocking=True).float()

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=cfg.AMP):
                        y = model(data)
                        loss = criterion(y)

                    if phase == "train":
                        if cfg.AMP:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                            optimizer.step()
                    torch.cuda.synchronize()
                    if cfg.MODEL_EMA:
                        model_ema.update(model)

                    hist_epoch_loss += loss * data.size(0)
                if rank in [-1, 0]:
                    progress_bar.set_description(
                        f"Epoch: {epoch + 1}/{max_epoch}. Loss: {loss.item():7.5f}"
                    )

            # Finish Train or Val Epoch Process below
            if rank != -1:
                # ここで，各プロセスのLossを全て足し合わせる
                # 正確なLossは全プロセスの勾配の平均を元にして計算するべき
                # https://discuss.pytorch.org/t/average-loss-in-dp-and-ddp/93306/8
                # reduceした時点で平均化されている
                dist.all_reduce(hist_epoch_loss, op=dist.ReduceOp.SUM)
                dist.barrier()
            epoch_loss = hist_epoch_loss.item() / len(datasets[phase])

            if phase == "train":
                if cfg.LR_SCHEDULER.NAME == "ReduceLROnPlateau":
                    scheduler.step(epoch_loss)
                elif cfg.LR_SCHEDULER.NAME == "CosineLRScheduler":
                    scheduler.step(epoch + 1)
                else:
                    scheduler.step()

            if rank in [-1, 0]:
                logger.info(
                    f"{phase.capitalize()} Epoch: {epoch + 1}/{max_epoch}. "
                    f"Loss: {epoch_loss:8.5f} "
                    f"GPU: {torch.cuda.memory_reserved(device) / 1e9:.1f}GB. "
                    f"Learning Rate: {optimizer.param_groups[0]['lr']:.4e}"
                )
                metric_values = [epoch_loss, optimizer.param_groups[0]["lr"]]
                writer.log_metrics(phase, metrics, metric_values, epoch + 1)

                if phase == "train":
                    # Save Model Weight
                    if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
                        save_model(model, save_model_path / f"model_epoch_{epoch+1}.pth")
                elif phase == "val":
                    # Save best val Loss Model
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_epoch = epoch
                        save_model(model, save_model_path / f"model_best_{epoch+1}.pth")
                        logger.info(
                            f"Save model at best val loss({best_loss:.4f}) in Epoch {best_epoch+1}"
                        )
                    # early stopping (check val_loss)
                    if epoch - best_epoch > int(cfg.EARLY_STOP_PATIENCE) > 0:
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
    return best_loss


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    # Set Local Rank for Multi GPU Training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Hydra Setting
    set_hydra(cfg, verbose=local_rank in [0, -1])

    if local_rank in [0, -1]:
        # make Output dir
        prefix = f"{cfg.MODEL.NAME}_{cfg.DATASET.NAME}"
        if cfg.TAG:
            prefix += f"_{cfg.TAG}"
        output_dir = make_output_dirs(
            cfg.OUTPUT,
            prefix=prefix,
            child_dirs=["figs", "models"],
        )

        # ClearML
        if cfg.USE_CLEARML:
            try:
                Task.init(project_name=cfg.EXPERIMENT_NAME, task_name=prefix)
            except Exception:
                logger.info("Not Installed ClearML")

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
    else:
        output_dir = ""
        setup_logger()
        writer = None

    # PyTorch A6000 Bug Fix: GPU間通信をP2PからPCI or NVLINKに変更する
    # os.environ["NCCL_P2P_DISABLE"] = "1"

    # set Device
    set_device(cfg.GPU.USE, is_cpu=cfg.CPU, verbose=local_rank in [0, -1])

    # DDP Mode
    if bool(cfg.GPU.MULTI):
        dist.init_process_group(backend="nccl", init_method="env://")
        logging.info(
            f"hostname={os.uname()[1]}, LOCAL_RANK={local_rank}, "
            f"RANK={dist.get_rank()}, WORLD_SIZE={dist.get_world_size()}"
        )

    try:
        result = do_train(local_rank, cfg, output_dir, writer)
    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"{e}\n{traceback.format_exc()}")
        if local_rank in [0, -1]:
            post_slack(
                channel="#error",
                message=f"Error\n{e}\n{traceback.format_exc()}\nOutput: {output_dir}",
            )
            writer.close("FAILED")
        if local_rank not in [0, -1]:
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
        if local_rank == 0:
            cfg.GPU.MULTI = False
            cfg.GPU.USE = 0
        OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))
        writer.log_artifact(os.path.join(output_dir, "config.yaml"))
        writer.log_artifacts(output_dir)

    # Clean Up multi gpu process
    if local_rank not in [0, -1]:
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
        result = do_test(cfg, output_result_dir, device, writer)
        message_dict["Test save"] = output_result_dir
        message_dict["result"] = result
        message = pprint.pformat(message_dict, width=150)
        # Send Message to Slack
        post_slack(message=f"Finish Test\n{message}")
        logger.info(f"Finish Test {message}")
        writer.log_artifacts(output_dir)
        writer.log_artifacts(output_result_dir)
        writer.close()
    return result


if __name__ == "__main__":
    main()
