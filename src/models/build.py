import logging
import torch
from torch.nn.parallel import DistributedDataParallel

from timm.utils import ModelEmaV2

from kunai import Registry

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
        model_ema = ModelEmaV2(model, decay=cfg.MODEL_EMA_DECAY)

    if rank != -1:
        if cfg.MODEL.SYNC_BN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=False
        )
        if cfg.MODEL_EMA:
            model_ema.set(model)
    elif torch.cuda.device_count() > 1 and phase == "train":
        logger.info("Use DataParallel Training")
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    return model, model_ema
