import logging

import torch
from kunai import Registry
from timm.utils import ModelEmaV2
from torch.nn.parallel import DistributedDataParallel

try:
    from timm.layers import convert_sync_batchnorm as convert_sync_batchnorm_timm
except Exception:
    from timm.models.layers import convert_sync_batchnorm as convert_sync_batchnorm_timm

MODEL_REGISTRY = Registry("MODEL")
# Get root logger
logger = logging.getLogger()


def build_model(cfg, device, phase="train", rank=-1):
    """build model

    Args:
        cfg (OmegaConf): Hydra Conf

    Returns:
        model: Torch.model
    """
    model_name = cfg.MODEL.MODEL
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model = model.to(device)

    model_ema = None
    if cfg.MODEL_EMA:
        logger.info("Use Model Exponential Moving Average(EMA)")
        model_ema = ModelEmaV2(model, decay=cfg.MODEL_EMA_DECAY)

    num_parameters, num_trainable_parameters, num_backbone_parameters = calc_model_prameters(model)
    logger.info(f"Num Model Parameters: {num_parameters}")
    logger.info(f"Num Trainable Model Parameters: {num_trainable_parameters}")
    logger.info(f"Num Backbone Model Parameters: {num_backbone_parameters}")

    if rank != -1:
        if cfg.MODEL.SYNC_BN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        elif cfg.MODEL.TIMM_SYNC_BN:
            logger.info("USE Timm Sync BatchNorm")
            model = convert_sync_batchnorm_timm(model)

        model = DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=cfg.MODEL.get("FIND_UNUSED_PARAMETERS", False),
        )
        if cfg.MODEL_EMA:
            model_ema.set(model)
    elif torch.cuda.device_count() > 1 and phase == "train":
        logger.info("Use DataParallel Training")
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    return model, model_ema


def calc_model_prameters(model: torch.nn.Module):
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
