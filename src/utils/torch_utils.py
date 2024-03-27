import os

import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel


def set_device(
    global_gpu_index,
    rank=-1,
    is_cpu=False,
    use_cudnn=True,
    cudnn_deterministic=False,
    pci_device_order=True,
    verbose=True,
):
    """Set use GPU or CPU Device

    set using GPU or CPU Device(instead of CUDA_VISIBLE_DEVICES).
    set also CUDNN.

    Args:
        global_gpu_index (int): using gpu number in all gpu.
        rank (int): process rank
        is_cpu (bool, optional): use cpu or not. Defaults to False.
        pci_device_order (bool, optional): . Defaults to True.

    Returns:
        torch.device: use device object.
    """

    if not is_cpu:
        if pci_device_order:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(global_gpu_index)

        # print using GPU Info
        if verbose:
            cuda_info(int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]))
            print(f"Using GPU is CUDA:{global_gpu_index}")

        if use_cudnn and cudnn.is_available():
            cudnn.benchmark = True
            cudnn.deterministic = cudnn_deterministic  # 乱数固定のため
            if verbose:
                print("Use CUDNN")
        if rank == -1:
            rank = 0
        device = torch.device(rank)
        torch.cuda.set_device(rank)
    else:
        device = torch.device("cpu")
        if verbose:
            print("Use CPU")

    return device


def cuda_info(global_cuda_index=0, logger=None):
    """show using GPU Info

    Args:
        global_cuda_index (int, optional): using GPU number in all GPU number. Defaults to 0.
    """

    for i in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(i)
        infostr = f"CUDA:{i + global_cuda_index} {info.name}, {info.total_memory / 1024 ** 2}MB"
        if logger is not None:
            logger.info(infostr)
        else:
            print(infostr)


def is_distributed():
    if dist.is_initialized():
        return True
    return False


def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor, n=1):
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


def adjust_learning_rate(base_lr, batch_size):
    world_size = get_world_size()
    return base_lr * batch_size * world_size


def check_model_parallel(model) -> bool:
    """check model is parallel or single

    Args:
        model (torch.nn.Module): Model file

    Returns:
        bool: parallel = True, single = False
    """
    return isinstance(model, torch.nn.DataParallel) or isinstance(model, DistributedDataParallel)


def load_model_weight(weight_path, model, logger=None):
    """Load PreTrained or Continued Model

    Args:
        model (torch.nn.Model): Load model
        weight (str): PreTrained weight path
    """
    if not weight_path:
        return

    if check_model_parallel(model):
        model = model.module
    device = next(model.parameters()).device
    model_state_dict = model.state_dict()

    check_point = torch.load(weight_path, map_location=device)
    # for torch.compile model
    checkpoint_state_dict = {}
    for k, v in check_point.items():
        checkpoint_state_dict[k.replace("_orig_mod.", "").replace("module.", "")] = v

    unmatch = []
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
                unmatch.append(k)

    missing, unexpexted = model.load_state_dict(checkpoint_state_dict, strict=False)

    if logger:
        logger.info(f"Load model weight from {weight_path}")
        logger.info(f"Missing model key: {missing}")
        logger.info(f"Unmatch model key: {unmatch}")
        logger.info(f"Unexpected model key: {unexpexted}")
