import logging

from kunai import Registry
from kunai.torch_utils import worker_init_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..transform import build_transforms
from .iteratable_dataloader import IterBasedDataloader

DATASET_REGISTRY = Registry("DATASET")

# Get root logger
logger = logging.getLogger()


def build_dataset(cfg, phase="train", rank=-1):
    transforms, batched_transform = build_transforms(cfg, phase=phase)
    dataset = DATASET_REGISTRY.get(cfg.DATASET.TYPE)(cfg, transforms, phase=phase)
    logger.info(f"{phase.capitalize()} Dataset sample num: {len(dataset)}")
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
