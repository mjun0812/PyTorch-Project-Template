import torch
import torch._dynamo as dynamo  # noqa
from timm.utils import ModelEmaV2
from torch.nn.parallel import DistributedDataParallel

from ..utils import Registry, load_model_weight

try:
    from timm.layers import convert_sync_batchnorm as convert_sync_batchnorm_timm
except Exception:
    from timm.models.layers import convert_sync_batchnorm as convert_sync_batchnorm_timm

MODEL_REGISTRY = Registry("MODEL")


def build_model(cfg, device, phase="train", rank=-1, logger=None):
    """build model

    Args:
        cfg (OmegaConf): Hydra Conf

    Returns:
        model: Torch.model
    """
    model_name = cfg.MODEL.MODEL
    model = MODEL_REGISTRY.get(model_name)(cfg, phase=phase)
    model = model.to(device)

    model_ema = None
    if cfg.MODEL_EMA:
        if logger is not None:
            logger.info("Use Model Exponential Moving Average(EMA)")
        model_ema = ModelEmaV2(model, decay=cfg.MODEL_EMA_DECAY)

    (
        num_parameters,
        num_trainable_parameters,
        num_backbone_parameters,
    ) = calc_model_prameters(model)
    if logger is not None:
        logger.info(f"Num Model Parameters: {num_parameters}")
        logger.info(f"Num Trainable Model Parameters: {num_trainable_parameters}")
        logger.info(f"Num Backbone Model Parameters: {num_backbone_parameters}")

    if rank != -1:
        if cfg.MODEL.SYNC_BN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        elif cfg.MODEL.TIMM_SYNC_BN:
            if logger is not None:
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
        if logger is not None:
            logger.info("Use DataParallel Training")
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    if cfg.CONTINUE_TRAIN:
        weight_path = cfg.MODEL.WEIGHT
    elif cfg.MODEL.PRE_TRAINED and cfg.MODEL.PRE_TRAINED_WEIGHT:
        weight_path = cfg.MODEL.PRE_TRAINED_WEIGHT
    else:
        weight_path = None
    load_model_weight(weight_path, model, logger)

    if cfg.COMPILE and torch.__version__ >= "2.0.0":
        dynamo.reset()
        model = torch.compile(model, backend=cfg.COMPILE_BACKEND)
        if logger is not None:
            logger.info("Use Torch Compile")

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
