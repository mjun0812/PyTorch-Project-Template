import os
import random
import time
from collections import OrderedDict
from logging import Logger
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torchinfo import summary

from ..types import PathLike


def worker_init_fn(worker_id: int) -> None:
    """Reset random seeds for PyTorch DataLoader workers.

    Ensures reproducible random number generation across different
    DataLoader worker processes.

    Args:
        worker_id: Unique identifier for the DataLoader worker.
    """
    # random
    random.seed(random.getstate()[1][0] + worker_id)
    # Numpy
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    # PyTorch
    torch.manual_seed(np.random.get_state()[1][0] + worker_id + 1)
    torch.cuda.manual_seed_all(np.random.get_state()[1][0] + worker_id)


def fix_seed(seed: int) -> int:
    """Set random seeds for reproducible results.

    Sets seeds for Python's random module, NumPy, and PyTorch
    to ensure reproducible results across runs.

    Args:
        seed: Random seed value.

    Returns:
        The seed value that was set.
    """
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed


def time_synchronized() -> float:
    """Return synchronized time between CUDA, MPS and CPU.

    Synchronizes CUDA/MPS operations with CPU before getting the current time
    to ensure accurate timing measurements.

    Returns:
        Current time after device synchronization.
    """
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    return time.time()


def setup_device(
    device_type: str,
    device_index: str | int = 0,
    rank: int = -1,
    use_cudnn: bool = True,
    cudnn_deterministic: bool = False,
    allow_tf32: bool = False,
    pci_device_order: bool = True,
    verbose: bool = True,
) -> torch.device:
    """Setup device from configuration with all necessary settings.

    Args:
        device_type: Device type ("cuda", "mps", "cpu").
        device_index: Device index for CUDA devices.
        rank: Process rank for distributed training.
        use_cudnn: Whether to use cuDNN backend.
        cudnn_deterministic: Whether to use cuDNN deterministic.
        allow_tf32: Whether to use TensorFloat-32 format.
        pci_device_order: Whether to use PCI device order.
        verbose: Whether to print verbose information.

    Returns:
        torch.device: Configured PyTorch device.
    """
    device_type = device_type.lower()

    if device_type == "cpu":
        device = torch.device("cpu")
        if verbose:
            print("Using CPU")
    elif device_type == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            if verbose:
                print("Using MPS (Metal Performance Shaders)")
        else:
            raise RuntimeError("MPS not available")
    elif device_type == "cuda":
        if torch.cuda.is_available():
            if pci_device_order:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

            if verbose:
                cuda_info(int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]))
                print(f"Using CUDA:{device_index}")

            if use_cudnn and cudnn.is_available():
                cudnn.benchmark = True
                cudnn.deterministic = cudnn_deterministic
                if verbose:
                    print("Using cuDNN")

            # Handle rank for distributed training
            if rank == -1:
                rank = 0
            device = torch.device(rank)
            torch.cuda.set_device(rank)
        else:
            raise RuntimeError("CUDA not available")
    else:
        raise ValueError(f"Unsupported device type: {device_type}")

    return device


def cuda_info(global_cuda_index: int = 0, logger: Logger | None = None) -> None:
    """Show using GPU Info

    Args:
        global_cuda_index (int, optional): using GPU number in all GPU number. Defaults to 0.
        logger (Logger, optional): Logger object. Defaults to None.
    """
    for i in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(i)
        infostr = f"CUDA:{i + global_cuda_index} {info.name}, {info.total_memory / 1024**2}MB"
        if logger is not None:
            logger.info(infostr)
        else:
            print(infostr)


def is_distributed() -> bool:
    return dist.is_initialized()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", -1))


def get_world_rank() -> int:
    return int(os.environ.get("RANK", -1))


def get_local_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


def is_local_main_process() -> bool:
    return not is_distributed() or get_local_rank() == 0


def is_world_main_process() -> bool:
    return not is_distributed() or get_world_rank() == 0


def is_multi_node() -> bool:
    return is_distributed() and (get_local_size() != get_world_size())


def reduce_tensor(tensor: torch.Tensor, n: int = 1) -> torch.Tensor:
    """分散学習時に，指定したtensorを各プロセスから集めて総和を取る

    Args:
        tensor (torch.Tensor): 総和を取るtensoe
        n (int, optional): 総和を割る数．word_size(GPU数)で割って平均を出すのに使う Defaults to 1.

    Returns:
        torch.Tensor: 結果
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    dist.barrier()
    return rt / n


def is_model_parallel(model: torch.nn.Module) -> bool:
    """Check model is parallel or single

    Args:
        model (torch.nn.Module): Model file

    Returns:
        bool: parallel = True, single = False
    """
    return isinstance(model, (torch.nn.DataParallel, DistributedDataParallel))


def is_model_compiled(model: torch.nn.Module) -> bool:
    """Check model is compiled or not using torch.compile

    Args:
        model (torch.nn.Module): Model object

    Returns:
        bool: compiled = True, not compiled = False
    """
    return hasattr(model, "_orig_mod")


def remove_compile_prefix_from_weight(state_dict: OrderedDict) -> OrderedDict:
    """torch.compileすると、重みに_orig_mod.がつくので削除

    Args:
        state_dict (dict): state_dict

    Returns:
        OrderedDict: state_dict
    """
    compile_prefix = "_orig_mod."
    keys = list(state_dict.keys())  # キーのリストを作成
    for k in keys:
        if k.startswith(compile_prefix):
            state_dict[k.replace(compile_prefix, "")] = state_dict.pop(k)
    return state_dict


def remove_parallel_prefix_from_weight(state_dict: OrderedDict) -> OrderedDict:
    """DataParallelすると、重みにmodule.がつくので削除

    Args:
        state_dict (OrderedDict): state_dict

    Returns:
        OrderedDict: state_dict
    """
    parallel_prefix = "module."
    keys = list(state_dict.keys())  # キーのリストを作成
    for k in keys:
        if k.startswith(parallel_prefix):
            state_dict[k.replace(parallel_prefix, "")] = state_dict.pop(k)
    return state_dict


def load_model_weight(weight_path: str | PathLike, model: torch.nn.Module) -> None:
    """Load PreTrained or Continued Model

    Args:
        weight_path (str): PreTrained weight path
        model (torch.nn.Module): Model object
    """
    if not weight_path:
        return

    # DDP + torch.compileの場合、_orig_mod.module.nameになるので、
    # _orig_mod　-> moduleの順に元モデルの参照を行う
    if is_model_compiled(model):
        model = model._orig_mod
    if is_model_parallel(model):
        model = model.module

    device = next(model.parameters()).device
    model_state_dict = model.state_dict()

    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    checkpoint = remove_compile_prefix_from_weight(checkpoint)
    checkpoint = remove_parallel_prefix_from_weight(checkpoint)

    unmatch = []
    for k in list(checkpoint.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint.pop(k)
                unmatch.append(k)

    missing, unexpexted = model.load_state_dict(checkpoint, strict=False)

    logger.info(f"Load model weight from {weight_path}")
    logger.info(f"Missing weight key: {missing}")
    logger.info(f"Unmatch weight's shape key: {unmatch}")
    logger.info(f"Unexpected weight key: {unexpexted}")


def save_model(model: torch.nn.Module, file_path: str | Path) -> None:
    if is_model_compiled(model):
        model = model._orig_mod
    if is_model_parallel(model):
        model = model.module
    state_dict = model.state_dict()  # For FSDP

    if is_world_main_process():
        torch.save(state_dict, str(file_path))
    logger.info(f"Saving model at {file_path!s}")


def save_optimizer(optimizer: torch.optim.Optimizer, file_path: str | Path) -> None:
    state_dict = optimizer.state_dict()
    if is_world_main_process():
        torch.save(state_dict, str(file_path))
    logger.info(f"Saving optimizer at {file_path!s}")


def save_lr_scheduler(
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler, file_path: str | Path
) -> None:
    torch.save(lr_scheduler.state_dict(), str(file_path))
    logger.info(f"Saving lr_scheduler at {file_path!s}")


def save_model_info(
    output_dir: PathLike | None,
    model: torch.nn.Module,
    input_size: list[int] | None = None,
    input_data: list[torch.Tensor] | None = None,
    prefix: str = "",
) -> str:
    """Output PyTorch Model Summary to log.

    Args:
        output_dir (string): output log dir
        model (torch.nn.Module): PyTorch Model Class
        input_size (List): input tensor size
        input_data (List[Tensor]): input data
        prefix (str, optional): log file prefix output_dir/model_summary_{prefix}.log.
                                Defaults to "".
    """
    if prefix:
        prefix = "_" + prefix
    if is_model_compiled(model):
        model = model._orig_mod
    if is_model_parallel(model):
        model = model.module

    device = next(model.parameters()).device

    if input_size is None:
        model_summary = str(summary(model, input_data=input_data, device=device, verbose=0))
    elif input_data is None:
        model_summary = str(summary(model, input_size=input_size, device=device, verbose=0))
    else:
        model_summary = str(summary(model, device=device, verbose=0))

    # Model Summary
    if output_dir is not None:
        with open(Path(output_dir) / f"model_summary{prefix}.log", "a") as f:
            print(model, file=f)
            print(model_summary, file=f)
    return model_summary
