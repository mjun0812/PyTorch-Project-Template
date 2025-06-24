import platform

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
"""Registry for dataset classes."""


def is_ram_cache_supported() -> bool:
    """Check if RAM cache is supported on the current platform.

    Returns:
        bool: True if RAM cache is supported (Linux only), False otherwise.
    """
    return platform.system() == "Linux"


def validate_ram_cache_config(use_ram_cache: bool, verbose: bool = True) -> bool:
    """Validate RAM cache configuration for current platform.

    Args:
        use_ram_cache: Whether RAM cache is requested.
        verbose: Whether to log warnings.

    Returns:
        bool: True if RAM cache can be used, False otherwise.
    """
    if use_ram_cache and not is_ram_cache_supported():
        if verbose:
            logger.warning(
                f"RAM cache is only supported on Linux. Current OS: {platform.system()}. "
                "Disabling RAM cache."
            )
        return False
    return use_ram_cache


def build_dataset(
    cfg: DatasetConfig,
    transforms: Compose | None,
    use_ram_cache: bool = False,
    ram_cache_size_gb: int | None = None,
) -> Dataset:
    """Build a dataset instance from configuration.

    Args:
        cfg: Dataset configuration containing class name and arguments.
        transforms: Optional data transformations to apply.
        use_ram_cache: Whether to enable RAM caching for improved performance.
        ram_cache_size_gb: Size limit for RAM cache in GB.

    Returns:
        Instantiated dataset with optional caching and transforms.

    Raises:
        AssertionError: If RAM cache size exceeds available shared memory.
        RuntimeError: If RAM cache is requested on non-Linux systems.
    """
    # Validate RAM cache configuration
    use_ram_cache = validate_ram_cache_config(use_ram_cache)

    if use_ram_cache and ram_cache_size_gb is not None:
        if ram_cache_size_gb > get_free_shm_size() / BYTES_PER_GIB:
            raise RuntimeError("RAM Cache size is too large")
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
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader from dataset and configuration.

    Args:
        dataset: PyTorch dataset to load data from.
        num_workers: Number of worker processes for data loading.
        batch_sampler: Batch sampler for organizing data into batches.
        use_iter_loop: Whether to use iteration-based training.
        max_iter: Maximum number of iterations for iter-based training.
        step_iter: Number of iterations per step for iter-based training.
        pin_memory: Whether to pin memory for faster data transfer.

    Returns:
        DataLoader instance, optionally wrapped for iteration-based training.
    """
    common_kwargs = {
        "pin_memory": pin_memory,
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
    """Build data samplers for training phases.

    Creates appropriate samplers based on the training phase and distributed
    training configuration.

    Args:
        dataset: PyTorch dataset to sample from.
        phase: Training phase (train, val, test).
        batch_size: Number of samples per batch.
        batch_sampler: Optional custom batch sampler name.

    Returns:
        Tuple of (sampler, batch_sampler) instances.
    """
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
    """Collate function for combining dataset outputs into batches.

    Automatically handles stacking of tensors and conversion of numeric
    values to tensors.

    Args:
        batch: List of dataset outputs to combine.

    Returns:
        Dictionary mapping keys to batched tensors.
    """
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
