import logging

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kunai.torch_utils import worker_init_fn

from . import Dataset
from ..transform import build_transforms

# Get root logger
logger = logging.getLogger()


def build_dataset(cfg, phase="train", rank=-1):
    if phase == "train":
        filelist = cfg.DATASET.TRAIN_LIST
    elif phase == "val":
        filelist = cfg.DATASET.VAL_LIST
    elif phase == "test":
        filelist = cfg.DATASET.TEST_LIST

    transform = build_transforms(cfg, phase=phase)
    dataset = Dataset(cfg, filelist)
    logger.info(f"{phase.capitalize()} Dataset sample num: {len(dataset)}")
    logger.info(f"{phase.capitalize()} transform: {transform}")

    common_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.NUM_WORKER,
        "batch_size": cfg.BATCH,
        "sampler": None,
        "worker_init_fn": worker_init_fn,
        "drop_last": True,
        "shuffle": True,
    }
    if rank != -1 and phase == "train":
        common_kwargs["shuffle"] = False
        common_kwargs["sampler"] = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, **common_kwargs)

    return dataset, dataloader