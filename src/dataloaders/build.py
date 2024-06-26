import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..transform import build_transforms
from ..utils import BYTES_PER_GIB, Registry, TensorCache, get_shm_size, worker_init_fn
from .iteratable_dataloader import IterBasedDataloader

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg, phase="train", rank=-1, logger=None):
    cfg_dataset = cfg.get(f"{phase.upper()}_DATASET")

    transforms, batched_transform = build_transforms(cfg, phase=phase)

    cache = None
    if phase == "train" and cfg.USE_RAM_CACHE:
        cache_size_gb = int(get_shm_size() / BYTES_PER_GIB) / 8
        cache = TensorCache(size_limit_gb=cache_size_gb)
        if logger is not None:
            logger.info(f"Use RAM Cache: {cache_size_gb}GB")
    dataset = DATASET_REGISTRY.get(cfg_dataset.TYPE)(cfg, transforms, phase=phase, cache=cache)

    if logger is not None:
        logger.info(f"{phase.capitalize()} {cfg_dataset.NAME} Dataset sample num: {len(dataset)}")
        logger.info(f"{phase.capitalize()} transform: {transforms}")
        if batched_transform is not None:
            logger.info(f"{phase.capitalize()} batched transform: {batched_transform}")

    common_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.NUM_WORKER,
        "batch_size": cfg.BATCH,
        "sampler": None,
        "worker_init_fn": worker_init_fn,
        "drop_last": phase == "train",
        "shuffle": phase == "train",
    }
    if rank != -1:
        common_kwargs["shuffle"] = False
        common_kwargs["sampler"] = DistributedSampler(dataset, shuffle=(phase == "train"))
    dataloader = DataLoader(dataset, **common_kwargs)

    if phase == "train" and cfg.ITER_TRAIN:
        max_iter = cfg.MAX_ITER
        step_iter = cfg.STEP_ITER
        dataloader = IterBasedDataloader(dataloader, max_iter, step_iter)

    return dataset, dataloader, batched_transform


def collate(batch):
    keys = list(batch[0].keys())
    output = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            output[k].append(b[k])
    for k in keys:
        if torch.is_tensor(output[k][0]):
            output[k] = torch.stack(output[k], dim=0)
    return output
