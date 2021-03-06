import glob
import logging
import os
import sys
import traceback
import pathlib
import pprint

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from natsort import natsorted

from kunai.hydra_utils import set_hydra
from kunai.torch_utils import fix_seed, save_model, save_model_info, set_device, worker_init_fn
from kunai.utils import get_cmd, get_git_hash, setup_logger, make_output_dirs

from src.dataloaders import Dataset
from src.losses import build_loss
from src.models import build_model
from src.transform import build_transforms
from src.utils import (
    TensorboardLogger,
    build_lr_scheduler,
    build_optimizer,
    plot_multi_graph,
    post_slack,
)


# Get root logger
logger = logging.getLogger()


def build_dataset(cfg, phase="train", rank=-1):
    if phase == "train":
        filelist = cfg.DATASET.TRAIN_LIST
    elif phase == "val":
        filelist = cfg.DATASET.VAL_LIST
    elif phase == "test":
        filelist = cfg.DATASET.TEST_LIST

    transform = build_transforms(cfg, phase=phase)
    dataset = Dataset(cfg, filelist)
    logger.info(f"{phase.capitalize()} Dataset sample num: {len(dataset)}")
    logger.info(f"{phase.capitalize()} transform: {transform}")

    common_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.NUM_WORKER,
        "batch_size": cfg.BATCH,
        "sampler": None,
        "worker_init_fn": worker_init_fn,
        "drop_last": True,
        "shuffle": True,
    }
    if rank != -1 and phase == "train":
        common_kwargs["shuffle"] = False
        common_kwargs["sampler"] = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, **common_kwargs)

    return dataset, dataloader


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
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module
        model.load_state_dict(torch.load(weight_path, map_location=device))
    except RuntimeError:
        # fine tuning
        logger.warning("Class num changed from loading weights. Do FineTuning?")

    logger.info(f"Load weight from {weight_path}")
    return last_epoch


def do_train(rank, cfg, output_dir):
    """Training Script
       ??????Function???Single GPU???Multi GPU????????????????????????????????????

    Args:
        rank (int): Process????????????Single GPU????????????-1???Multi GPU????????????0,1...
                    ????????????Multi???Single????????????Master Prosess??????????????????????????????
                    MultiGPU????????????MasterProcess?????????File IO???????????????NG?????????
                    ????????????FileIO???????????????????????????
                    rank = [-1,0]????????????IO??????????????????????????????IO??????????????????
                    IO?????????????????????????????????????????????????????????
        cfg (OmegaConf): Hydra Conf
        output_dict (dict): result???????????????Directry Path
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

    # ###### Build Model #######
    model = build_model(cfg).to(device)
    if rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=False
        )
    # Train from exist weight
    last_epoch = load_last_weight(cfg, model)

    # ####### Build Dataset and Dataloader #######
    logger.info("Loading Dataset...")
    datasets = {}
    dataloaders = {}
    for phase in ["train", "val"]:
        datasets[phase], dataloaders[phase] = build_dataset(cfg, phase=phase, rank=rank)
    logger.info("Complete Loading Dataset")

    # ###### Logging ######
    histories = {}
    metrics = ["Loss", "Learning Rate"]
    metrics_dict = {metric: [] for metric in metrics}
    for phase in ["train", "val"]:
        histories[phase] = metrics_dict
    if rank in [-1, 0]:
        # Model???????????????
        save_model_info(output_dir, model)
        # save initial model
        save_model(model, save_model_path / f"model_init_{last_epoch}.pth")
        # Set Tensorboard
        writer = TensorboardLogger(output_dir)
        if cfg.MODEL.GRAPH and rank == -1:
            writer.write_model_graph(cfg, model, device)

    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    max_epoch = cfg.EPOCH + last_epoch
    best_loss = 1e8
    best_epoch = 0

    try:
        logger.info("Start Training")
        # Train Loop
        for epoch in range(last_epoch, max_epoch, 1):
            logger.info(f"Start Epoch {epoch+1}")
            np.random.seed(initial_seed + epoch + rank)
            for phase in ["train", "val"]:
                hist_epoch_loss = 0

                # Skip Validation
                if phase == "val" and ((epoch + 1) % cfg.VAL_INTERVAL != 0):
                    continue

                # model.train(False) == model.eval()
                model.train(phase == "train")
                if phase == "train" and rank != -1:
                    dataloaders[phase].sampler.set_epoch(epoch)

                # Set progress bar
                progress_bar = enumerate(dataloaders[phase])
                if rank in [-1, 0]:
                    progress_bar = tqdm(progress_bar, total=len(dataloaders[phase]))

                for _, data in progress_bar:
                    with torch.set_grad_enabled(phase == "train"), torch.cuda.amp.autocast(
                        enabled=cfg.AMP
                    ):
                        data = data.to(device, non_blocking=True).float()

                        optimizer.zero_grad()

                        # Calculate Loss
                        y = model(data)
                        loss = criterion(y)
                        if loss == 0 or not torch.isfinite(loss):
                            continue
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                        hist_epoch_loss += loss * data.size(0)
                    if rank in [-1, 0]:
                        progress_bar.set_description(
                            f"Epoch: {epoch + 1}/{max_epoch}. Loss: {loss.item():7.5f}"
                        )

                # Finish Train or Val Epoch Process below
                if rank != -1:
                    # ??????????????????????????????Loss???????????????????????????
                    # ?????????Loss?????????????????????????????????????????????????????????????????????
                    # https://discuss.pytorch.org/t/average-loss-in-dp-and-ddp/93306/8
                    # reduce???????????????????????????????????????
                    dist.all_reduce(hist_epoch_loss, op=dist.ReduceOp.SUM)
                    dist.barrier()
                epoch_loss = hist_epoch_loss.item() / len(datasets[phase])
                if phase == "train":
                    if cfg.LR_SCHEDULER == "reduce":
                        scheduler.step(epoch_loss)
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
                    for metric, value in zip(metrics, metric_values):
                        writer.write_scalars(metric, {phase: value}, epoch + 1)
                        histories[phase][metric].append(value)

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
    except (Exception, KeyboardInterrupt) as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        if rank in [0, -1]:
            post_slack(channel="#error", message=f"Error\n{e}\n{traceback.format_exc()}")
        if rank != -1:
            dist.destroy_process_group()
        sys.exit(1)

    # Finish Training Process below
    if rank in [-1, 0]:
        writer.writer_close()
        save_model(model, os.path.join(save_model_path, f"model_final_{max_epoch}.pth"))

        try:
            # ?????????????????????????????????????????????????????????
            # Validation Interval???????????????len(hist_loss) > len(hist_val_loss)
            # ????????????x??????????????????????????????????????????????????????
            x = np.arange(len(histories["train"]["Loss"]))
            val_x = np.linspace(0, len(histories["train"]["Loss"]), len(histories["val"]["Loss"]))
            for metric in metrics:
                histories["val"][metric] = np.interp(x, val_x, histories["val"][metric])

            plot_data = {}
            for title in metrics:
                plot_data[title] = []
                for phase in ["train", "val"]:
                    if len(histories[phase][title]) > 0:
                        plot_data[title].append(
                            {"data": histories[phase][title], "label": phase.capitalize()}
                        )
            plot_multi_graph(
                os.path.join(output_dir, "figs", "train_graph.png"), plot_data.keys(), plot_data
            )
        except Exception:
            logger.error("Cannot draw graph")
    return best_loss


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    # Set Local Rank for Multi GPU Training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Hydra Setting
    set_hydra(cfg, verbose=local_rank in [0, -1])
    if local_rank in [0, -1]:
        prefix = f"{cfg.MODEL.NAME}_{cfg.DATASET.NAME}"
        if cfg.TAG:
            prefix += f"_{cfg.TAG}"
        output_dir = make_output_dirs(
            cfg.OUTPUT_PATH,
            prefix=prefix,
            child_dirs=["logs", "tensorboard", "figs", "models"],
        )
    else:
        output_dir = ""

    # Logging
    setup_logger(local_rank, os.path.join(output_dir, "train.log"))
    logger.info(f"Command: {get_cmd()}")
    logger.info(f"Make output_dir at {output_dir}")
    logger.info(f"Git Hash: {get_git_hash()}")

    # Save Info
    if local_rank in [0, -1]:
        OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))
        with open(os.path.join(output_dir, "cmd_histry.log"), "a") as f:
            print(get_cmd(), file=f)

    # set Device
    # PyTorch A6000 Bug Fix
    # GPU????????????P2P??????PCI or NVLINK???????????????
    os.environ["NCCL_P2P_DISABLE"] = "1"
    set_device(cfg.GPU.USE, is_cpu=cfg.CPU, verbose=local_rank in [0, -1])

    # DDP Mode
    if bool(cfg.GPU.MULTI):
        assert (
            torch.cuda.device_count() > 1
        ), f"plz check gpu num. current gpu num: {torch.cuda.device_count()}"
        dist.init_process_group(backend="nccl", init_method="env://")
        logging.info(
            f"hostname={os.uname()[1]}, LOCAL_RANK={local_rank}, "
            f"RANK={dist.get_rank()}, WORLD_SIZE={dist.get_world_size()}"
        )

    result = do_train(local_rank, cfg, output_dir)

    if local_rank in [0, -1]:
        message = pprint.pformat({
            "host": os.uname()[1],
            "tag": cfg.TAG,
            "model": cfg.MODEL.NAME,
            "dataset": cfg.DATASET.NAME,
            "save": output_dir,
            "test_cmd": f"python test.py -cp {output_dir}",
        }, width=1000)
        # Send Message to Slack
        post_slack(message=f"Finish Training\n{message}")
        logger.info(f"Finish Training {message}")

        # Prepare config for Test
        cfg.MODEL.WEIGHT = natsorted(
            glob.glob(os.path.join(output_dir, "models", "model_best_*.pth"))
        )[-1]
        if local_rank==0:
            cfg.GPU.MULTI=True
            cfg.GPU.USE=0
        OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    # Clean Up multi gpu process
    if local_rank == 0:
        dist.destroy_process_group()

    return result


if __name__ == "__main__":
    main()
