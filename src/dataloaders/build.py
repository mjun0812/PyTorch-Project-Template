import torch
from loguru import logger
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torchvision.transforms.v2 import Compose

from ..config import DatasetConfig
from ..types import PhaseStr
from ..utils import Registry, get_free_shm_size, is_distributed, worker_init_fn
from .base import BaseDataset
from .iteratable_dataloader import IterBasedDataloader
from .sampler import SAMPLER_REGISTRY
from .tensor_cache import BYTES_PER_GIB, TensorCache
from .types import DatasetOutput

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(
    cfg: DatasetConfig,
    transforms: Compose | None,
    use_ram_cache: bool = False,
    ram_cache_size_gb: int | None = None,
) -> Dataset:
    if use_ram_cache:
        assert (
            ram_cache_size_gb <= get_free_shm_size() / BYTES_PER_GIB
        ), "RAM Cache size is too large"
        cache = TensorCache(size_limit_gb=ram_cache_size_gb)
        logger.info(f"Use RAM Cache: {ram_cache_size_gb}GB")
    else:
        cache = None

    dataset_cls: BaseDataset = DATASET_REGISTRY.get(cfg.class_name)
    dataset = dataset_cls(cfg.args, transforms=transforms, cache=cache)
    return dataset


def build_dataloader(
    dataset: Dataset,
    num_workers: int,
    batch_sampler: BatchSampler,
    use_iter_loop: bool = False,
    max_iter: int | None = None,
    step_iter: int | None = None,
) -> DataLoader:
    common_kwargs = {
        "pin_memory": True,
        "num_workers": num_workers,
        "batch_sampler": batch_sampler,
        "worker_init_fn": worker_init_fn,
        "collate_fn": collate,
    }
    dataloader = DataLoader(dataset, **common_kwargs)

    if use_iter_loop:
        dataloader = IterBasedDataloader(dataloader, max_iter, step_iter)
    return dataloader


def build_sampler(
    dataset: Dataset,
    phase: PhaseStr = "train",
    batch_size: int = 32,
    batch_sampler: str | None = None,
) -> tuple[Sampler, BatchSampler]:
    # Build Sampler
    if is_distributed():
        sampler = DistributedSampler(dataset, shuffle=(phase == "train"))
    elif phase == "train":
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    if batch_sampler is not None:
        batch_sampler = SAMPLER_REGISTRY.get(batch_sampler)(
            sampler, batch_size, drop_last=(phase == "train")
        )
    else:
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=(phase == "train"))

    return sampler, batch_sampler


def collate(batch: list[DatasetOutput]) -> dict[str, torch.Tensor]:
    keys = list(batch[0].keys())
    output = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            output[k].append(b[k])
    for k in keys:
        if torch.is_tensor(output[k][0]):
            output[k] = torch.stack(output[k], dim=0)
        elif isinstance(output[k][0], (int, float)):
            output[k] = torch.tensor(output[k])
    return output
