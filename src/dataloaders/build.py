import logging

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kunai.torch_utils import worker_init_fn
from kunai import Registry

from ..transform import build_transforms

DATASET_REGISTRY = Registry("DATASET")

# Get root logger
logger = logging.getLogger()


def build_dataset(cfg, phase="train", rank=-1):
    transforms = build_transforms(cfg, phase=phase)
    dataset = DATASET_REGISTRY.get(cfg.DATASET.TYPE)(cfg, transforms, phase=phase)
    logger.info(f"{phase.capitalize()} Dataset sample num: {len(dataset)}")
    logger.info(f"{phase.capitalize()} transform: {transforms}")

    common_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.NUM_WORKER,
        "batch_size": cfg.BATCH,
        "sampler": None,
        "worker_init_fn": worker_init_fn,
        "drop_last": True,
        "shuffle": True,
    }
    if rank != -1:
        common_kwargs["shuffle"] = False
        common_kwargs["sampler"] = DistributedSampler(dataset, shuffle=(phase == "train"))
    dataloader = DataLoader(dataset, **common_kwargs)

    return dataset, dataloader
