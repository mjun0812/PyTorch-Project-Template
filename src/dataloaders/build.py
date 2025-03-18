from typing import Any, Optional

import torch
from loguru import logger
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler

from ..config import DatasetConfig, ExperimentConfig
from ..transform import build_transforms
from ..types import DatasetOutput, PhaseStr
from ..utils import Registry, get_free_shm_size, is_distributed, worker_init_fn
from .iteratable_dataloader import IterBasedDataloader
from .sampler import SAMPLER_REGISTRY
from .tensor_cache import BYTES_PER_GIB, TensorCache

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(
    cfg: ExperimentConfig, phase: PhaseStr = "train"
) -> tuple[Dataset, DataLoader, Any, Optional[Sampler]]:
    transforms, batched_transform = build_transforms(cfg, phase)

    # Build RAM Cache
    cache = None
    if phase == "train" and cfg.use_ram_cache:
        assert cfg.ram_cache_size_gb <= get_free_shm_size() / BYTES_PER_GIB, (
            "RAM Cache size is too large"
        )
        cache = TensorCache(size_limit_gb=cfg.ram_cache_size_gb)
        logger.info(f"Use RAM Cache: {cfg.ram_cache_size_gb}GB")

    # Build Dataset
    cfg_dataset: DatasetConfig = cfg.get(f"{phase}_dataset")
    dataset = DATASET_REGISTRY.get(cfg_dataset.dataset)(cfg, transforms, phase=phase, cache=cache)
    phase_cap = phase.capitalize()
    logger.info(f"{phase_cap} {cfg_dataset.name} Dataset sample num: {len(dataset)}")
    logger.info(f"{phase_cap} transform: {transforms}")
    if batched_transform is not None:
        logger.info(f"{phase_cap} batched transform: {batched_transform}")

    # Build Sampler
    if is_distributed():
        sampler = DistributedSampler(dataset, shuffle=(phase == "train"))
    elif phase == "train":
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    if cfg.batch_sampler is not None:
        batch_sampler = SAMPLER_REGISTRY.get(cfg.batch_sampler)(
            sampler, cfg.batch, drop_last=(phase == "train")
        )
    else:
        batch_sampler = BatchSampler(sampler, cfg.batch, drop_last=(phase == "train"))

    # Build Dataloader
    common_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.num_worker,
        "batch_sampler": batch_sampler,
        "worker_init_fn": worker_init_fn,
        "collate_fn": collate,
        # BatchSamplerを使用する場合は以下のキーワード引数は不要
        # "sampler": None,
        # "batch_size": cfg.batch,
        # "drop_last": phase == "train",
        # "shuffle": None,
    }
    dataloader = DataLoader(dataset, **common_kwargs)
    if phase == "train" and cfg.use_iter_loop:
        dataloader = IterBasedDataloader(dataloader, cfg.max_iter, cfg.step_iter)

    return dataset, dataloader, batched_transform, sampler


def collate(batch: list[DatasetOutput]) -> dict[str, torch.Tensor]:
    keys = list(batch[0].keys())
    output = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            output[k].append(b[k])
    for k in keys:
        if torch.is_tensor(output[k][0]):
            output[k] = torch.stack(output[k], dim=0)
    return output
