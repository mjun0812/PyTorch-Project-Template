import glob
import logging
import os
import shutil
import sys
import traceback
import pathlib

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

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


def build_dataset(cfg, phase="train"):
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
    return dataset


def init_process(rank, num_gpus, fn, fn_kwargs):
    """Initialize Process when multi GPU Training

    Args:
        rank (int): process rank
        num_gpus (int): number of gpus
        fn (function): training finction
        fn_kwargs (dict): training arguments
    """
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=num_gpus,
    )
    print(
        f"hostname={os.uname()[1]}, LOCAL_RANK={rank}, "
        f"RANK={dist.get_rank()}, WORLD_SIZE={dist.get_world_size()}"
    )
    fn_kwargs["device"] = torch.device(rank)
    setup_logger(rank, os.path.join(fn_kwargs["output_dir"], "train.log"))
    fn(rank, **fn_kwargs)


def load_last_weight(model, weight):
    """Load PreTrained or Continued Model

    Args:
        model (torch.nn.Model): Load model
        weight (str): PreTrained weight path

    Returns:
        model: Loaded Model
        last_epoch: number of last epoch from weight file name
                    ex. 'weight/model_epoch_15.pth' return '15'
    """
    logger.info(f"Load weight from {weight}")

    try:
        # If continue train, get final Epoch number
        last_epoch = os.path.splitext(weight)[0].split("_")[-1]
        last_epoch = int(last_epoch)
    except Exception:
        # Load Pretrained Weight
        last_epoch = 0

    # get device from model
    device = next(model.parameters()).device
    try:
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, DDP):
            model = model.module
        model.load_state_dict(torch.load(weight, map_location=device))
    except RuntimeError:
        # fine tuning
        logger.warning("Class num changed from loading weights. Do Fine Tuning?")
    return model, last_epoch


def do_train(rank, cfg, output_dir, device, num_gpus=1):
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
        device (Torch.device): CPU, Single GPUのときに利用
        num_gpus (int, optional): Multi　GPUのときに利用．GPUの数. Defaults to 1.
    """

    save_model_path = pathlib.Path(output_dir, "models")

    # ###### Build Model #######
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    if rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    # Train from Trained weight
    last_epoch = 0
    if cfg.CONTINUE_TRAIN:
        model, last_epoch = load_last_weight(model, cfg.MODEL.WEIGHT)
    # Train from Pretrained weight
    elif cfg.MODEL.PRE_TRAINED and cfg.MODEL.PRE_TRAINED_WEIGHT:
        model, _ = load_last_weight(model, cfg.MODEL.PRE_TRAINED_WEIGHT)
    if rank in [-1, 0]:
        # Model構造を出力
        save_model_info(
            output_dir,
            model,
            input_size=(1, 3, cfg.MODEL.INPUT_SIZE, cfg.MODEL.INPUT_SIZE),
        )
        # save initial model
        save_model(model, save_model_path / f"model_init_{last_epoch}.pth")
        # Tensorboardのセットアップ
        writer = TensorboardLogger(output_dir)
        if cfg.MODEL.GRAPH and rank == -1:
            writer.write_model_graph(cfg, model, device)
    else:
        writer = None

    # ####### Build Dataset and Dataloader #######
    logger.info("Loading Dataset...")
    common_kwargs = {
        "pin_memory": True,
        "num_workers": 4,
        "batch_size": cfg.BATCH,
        "sampler": None,
        "worker_init_fn": worker_init_fn,
        "drop_last": True,
    }
    datasets = {}
    dataloaders = {}
    for phase in ["train", "val"]:
        datasets[phase] = build_dataset(cfg, phase=phase)
        if rank != -1:
            common_kwargs["sampler"] = DistributedSampler(
                datasets["phase"], rank=rank, num_replicas=num_gpus, shuffle=True
            )
        dataloaders[phase] = DataLoader(datasets[phase], **common_kwargs)
    logger.info("Complete Loading Dataset")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    max_epoch = cfg.EPOCH + last_epoch
    best_loss = 1e5
    best_epoch = 0

    histories = {}
    for phase in ["train", "val"]:
        histories[phase] = {"Loss": [], "Learning Rate": []}

    if rank in [-1, 0]:
        logger.info("Start Training")

    initial_seed = fix_seed(100 + rank)
    try:
        # Train Loop
        for epoch in range(last_epoch, max_epoch, 1):
            for phase in ["train", "val"]:
                hist_epoch_loss = 0
                np.random.seed(initial_seed + epoch + rank)
                progress_bar = enumerate(dataloaders[phase])

                if phase == "train":
                    model.train()
                    if rank in [-1, 0]:
                        logger.info(f"Start Epoch {epoch+1}")
                        progress_bar = tqdm(progress_bar, total=len(dataloaders[phase]))
                elif (phase == "val") and (rank in [-1, 0]) and ((epoch + 1) % cfg.VAL_INTERVAL == 0):
                    model.eval()
                else:
                    continue

                if rank != -1:
                    dataloaders[phase].sampler.set_epoch(epoch)

                for _, data in progress_bar:
                    with torch.set_grad_enabled(phase == "train"):
                        data = data.to(device, non_blocking=True).float()

                        optimizer.zero_grad()

                        # Calculate Loss
                        y = model(data)
                        loss = loss_fn(y)
                        if loss == 0 or not torch.isfinite(loss):
                            continue
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    hist_epoch_loss += loss * data.size(0)
                    if rank in [-1, 0]:
                        progress_bar.set_description(
                            f"Epoch: {epoch + 1}/{max_epoch}. Loss: {loss.item():.5f}"
                        )

                # Finish Epoch Process below
                if rank != -1:
                    dist.all_reduce(hist_epoch_loss, op=dist.ReduceOp.SUM)
                    dist.barrier()
                epoch_loss = hist_epoch_loss.item() / len(datasets[phase])

                if rank in [-1, 0]:
                    logger.info(
                        f"{phase.capitalize()} Epoch: {epoch + 1}/{max_epoch}. "
                        f"Loss: {epoch_loss:.5f} "
                        f"GPU: {torch.cuda.memory_reserved(device) / 1e9:.1f}GB. "
                    )

                    if phase == "train":
                        scheduler.step(epoch_loss)
                        lr = optimizer.param_groups[0]["lr"]
                        writer.write_scalar("Learning_Rate", lr, epoch + 1)
                        histories[phase]["Learning Rate"].append(lr)
                    writer.write_scalars("Epoch_Loss", {phase: epoch_loss}, epoch + 1)
                    histories[phase]["Loss"].append(epoch_loss)

                    # Save Model Weight
                    if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
                        save_model(model, save_model_path / f"model_epoch_{epoch+1}.pth")

                    # Validate
                    if phase == "val":
                        # Save best val Loss Model
                        if loss < best_loss:
                            best_loss = loss
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
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        post_slack(message=f"Error\n{e}\n{traceback.format_exc()}")
        # Train中のエラーはディレクトリごと削除
        # Non fileやCUDA out of memoryなどのエラー発生時の時
        shutil.rmtree(output_dir)
        if rank != 0:
            dist.destroy_process_group()
        sys.exit(1)

    # Clean Up multi gpu process
    if rank != 0:
        dist.destroy_process_group()

    # Finish Training Process below
    if rank in [-1, 0]:
        writer.writer_close()
        try:
            # サンプル不足でのグラフ描画エラーの処理
            # Validation Intervalによってはlen(hist_loss) > len(hist_val_loss)
            # なので、x軸を補間することで、グラフを合わせる
            x = np.arange(len(histories["train"]["Loss"]))
            val_x = np.linspace(0, len(histories["train"]["Loss"]), len(histories["val"]["Loss"]))
            histories["val"]["Loss"] = np.interp(x, val_x, histories["val"]["Loss"])

            plot_data = {}
            for title in ["Loss", "Learning Rate"]:
                plot_data[title] = []
                for phase in ["train", "val"]:
                    if histories[phase][title]:
                        plot_data[title].append(
                            {"data": histories[phase][title], "label": phase.capitalize()}
                        )
            plot_multi_graph(os.path.join(output_dir, "figs", "train_graph.png"), data.keys(), data)
        except Exception:
            logger.error("Cannot draw graph")

        save_model(model, os.path.join(save_model_path, f"model_final_{max_epoch}.pth"))

    return best_loss


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    # Hydra Setting
    set_hydra(cfg)
    output_dir = make_output_dirs(
        cfg.OUTPUT_PATH,
        prefix=f"{cfg.MODEL.NAME}_{cfg.DATASET.NAME}",
        child_dirs=["logs", "tensorboard", "figs", "models"],
    )
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    # Logging
    setup_logger(-1, os.path.join(output_dir, "train.log"))
    logger.info(f"Command: {get_cmd()}")
    logger.info(f"Make output_dir at {output_dir}")
    logger.info(f"Git Hash: {get_git_hash()}")
    with open(os.path.join(output_dir, "cmd_histry.log"), "a") as f:
        print(get_cmd(), file=f)

    # set CPU or GPU Device
    device = set_device(cfg.GPU.USE, is_cpu=cfg.CPU)
    num_gpus = torch.cuda.device_count()

    # DDP Mode
    if bool(cfg.GPU.MULTI):
        assert num_gpus > 1, f"plz check gpu num. current gpu num: {num_gpus}"
        # Master Process's IP and Port.
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        mp.spawn(
            init_process,
            args=(
                num_gpus,
                do_train,
                {
                    "cfg": cfg,
                    "output_dir": output_dir,
                    "device": device,
                    "num_gpus": num_gpus,
                },
            ),
            nprocs=num_gpus,
        )
        result = 0
    # Single GPU Mode
    else:
        result = do_train(-1, cfg, output_dir, device)

    message = {
        "host": os.uname()[1],
        "tag": cfg.TAG,
        "model": cfg.MODEL.NAME,
        "dataset": cfg.DATASET.NAME,
        "save": output_dir,
        "test_cmd": f"python test.py -cp {output_dir}",
    }
    # Send Message to Slack
    post_slack(message=f"Finish Training\n{message}")
    logger.info(f"Finish Training {message}")

    cfg.MODEL.WEIGHT = glob.glob(os.path.join(output_dir, "models", "model_final_*.pth"))[0]
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    return result


if __name__ == "__main__":
    main()
