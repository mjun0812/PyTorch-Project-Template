from torch import distributed as dist


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
