import inspect

import kornia.augmentation as K
import torchvision.transforms.v2 as T
from omegaconf import OmegaConf

from ..config import ExperimentConfig, TransformConfig
from ..types import PhaseStr
from ..utils import Registry
from .compose import BatchedTransformCompose

TRANSFORM_REGISTRY = Registry("TRANSFORM")
BATCHED_TRANSFORM_REGISTRY = Registry("BATCHED_TRANSFORM")
for augmentation in inspect.getmembers(K, inspect.isclass):
    BATCHED_TRANSFORM_REGISTRY.register(augmentation[1])


def build_transforms(
    cfg: ExperimentConfig, phase: PhaseStr = "train"
) -> tuple[T.Compose, BatchedTransformCompose]:
    cfg_batched = None
    if phase == "train":
        cfg_transforms = cfg.train_dataset.train_transforms
        cfg_batched = cfg.train_dataset.train_batch_transforms
    elif phase == "val":
        cfg_transforms = cfg.val_dataset.val_transforms
    elif phase == "test":
        cfg_transforms = cfg.test_dataset.test_transforms

    transes = []
    for trans in cfg_transforms:
        transes.append(get_transform(trans))

    batched_transform = None
    if cfg_batched is not None:
        batched_transform = BatchedTransformCompose(cfg_batched)
    return T.Compose(transes), batched_transform


def get_transform(cfg: TransformConfig):
    if cfg.args is None:
        transform = TRANSFORM_REGISTRY.get(cfg.name)()
    else:
        transform = TRANSFORM_REGISTRY.get(cfg.name)(**OmegaConf.to_object(cfg.args))
    return transform
