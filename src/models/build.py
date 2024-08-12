from typing import NotRequired, TypedDict

import torch
import torch._dynamo as dynamo  # noqa
import torch.nn as nn
from loguru import logger
from timm.utils import ModelEmaV2
from torch.nn.parallel import DistributedDataParallel

from ..alias import PhaseStr
from ..config import ExperimentConfig
from ..losses import LossOutput, build_loss
from ..utils import Registry, get_local_rank, is_distributed, load_model_weight

try:
    from timm.layers import convert_sync_batchnorm as convert_sync_batchnorm_timm
except Exception:
    from timm.models.layers import convert_sync_batchnorm as convert_sync_batchnorm_timm


MODEL_REGISTRY = Registry("MODEL")


class ModelOutput(TypedDict, total=False):
    losses: NotRequired[LossOutput]
    preds: NotRequired[dict[str, torch.Tensor]]


class BaseModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig, phase: PhaseStr = "train"):
        super().__init__()
        self.cfg = cfg
        self.phase = phase

        self.loss = None
        if self.phase in ["train", "val"]:
            self.loss = build_loss(self.cfg)

    def train_forward(self, data: dict) -> ModelOutput:
        raise NotImplementedError

    def val_forward(self, data: dict) -> ModelOutput:
        raise NotImplementedError

    def test_forward(self, data: dict) -> ModelOutput:
        raise NotImplementedError

    def forward(self, data: dict) -> ModelOutput:
        if self.phase == "train":
            return self.train_forward(data)
        elif self.phase == "val":
            return self.val_forward(data)
        else:
            return self.test_forward(data)


def build_model(
    cfg: ExperimentConfig, device: torch.device, phase: PhaseStr = "train"
) -> tuple[BaseModel, torch.nn.Module]:
    """build model

    Args:
        cfg (OmegaConf): Hydra Conf

    Returns:
        model: Torch.model
    """
    model = MODEL_REGISTRY.get(cfg.model.model)(cfg, phase=phase)
    model = model.to(device)

    model_ema = None
    if cfg.model.use_model_ema:
        if logger is not None:
            logger.info("Use Model Exponential Moving Average(EMA)")
        model_ema = ModelEmaV2(model, decay=cfg.model.model_ema_decay)

    (
        num_parameters,
        num_trainable_parameters,
        num_backbone_parameters,
    ) = calc_model_prameters(model)
    logger.info(f"Num Model Parameters: {num_parameters}")
    logger.info(f"Num Trainable Model Parameters: {num_trainable_parameters}")
    logger.info(f"Num Backbone Model Parameters: {num_backbone_parameters}")

    if is_distributed():
        if cfg.model.use_sync_bn:
            model = convert_sync_batchnorm_timm(model)
            logger.info("USE Sync BatchNorm")

        model = DistributedDataParallel(
            model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=cfg.model.find_unused_parameters,
        )
        if cfg.model.use_model_ema:
            model_ema.set(model)
    elif torch.cuda.device_count() > 1 and phase == "train":
        if logger is not None:
            logger.info("Use DataParallel Training")
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    if cfg.model.pre_trained_weight:
        load_model_weight(cfg.model.pre_trained_weight, model)

    if cfg.use_compile and torch.__version__ >= "2.0.0":
        dynamo.reset()
        model = torch.compile(model, backend=cfg.compile_backend)
        logger.info("Use Torch Compile")

    return model, model_ema


def calc_model_prameters(model: torch.nn.Module) -> tuple[int, int, int]:
    num_params = 0
    num_backbone_prams = 0
    num_trainable_prams = 0
    for n, m in model.named_parameters():
        if "backbone" in n:
            num_backbone_prams += m.numel()
        if m.requires_grad:
            num_trainable_prams += m.numel()
        num_params += m.numel()
    return num_params, num_trainable_prams, num_backbone_prams
