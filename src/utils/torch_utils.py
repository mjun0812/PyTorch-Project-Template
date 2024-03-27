import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel


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
    world_size = 1
    if is_distributed():
        world_size = dist.get_world_size()
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
