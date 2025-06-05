from functools import partial

from loguru import logger
from timm.utils import ModelEmaV3
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel

from ..config import ExperimentConfig, ModelConfig
from ..types import TORCH_DTYPE, PhaseStr
from ..utils import Registry, get_local_rank, load_model_weight
from .base import BaseModel

try:
    from timm.layers import convert_sync_batchnorm
except ImportError:
    from timm.models.layers import convert_sync_batchnorm

MODEL_REGISTRY = Registry("MODEL")


def build_model(cfg: ModelConfig, phase: PhaseStr = "train") -> BaseModel:
    model = MODEL_REGISTRY.get(cfg.class_name)(cfg, phase)
    if cfg.pre_trained_weight:
        load_model_weight(cfg.pre_trained_weight, model)
    return model


def calc_model_parameters(model: nn.Module) -> tuple[int, int, int]:
    num_params = num_backbone_params = num_trainable_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        num_params += param_count
        if "backbone" in name:
            num_backbone_params += param_count
        if param.requires_grad:
            num_trainable_params += param_count
    return num_params, num_trainable_params, num_backbone_params


def log_model_parameters(model: nn.Module) -> None:
    num_params, num_trainable_params, num_backbone_params = calc_model_parameters(model)
    logger.info(f"Num Model Parameters: {num_params}")
    logger.info(f"Num Trainable Model Parameters: {num_trainable_params}")
    if num_backbone_params > 0:
        logger.info(f"Num Backbone Model Parameters: {num_backbone_params}")


def create_model_ema(cfg: ExperimentConfig, model: nn.Module) -> ModelEmaV3:
    logger.info("Use Model Exponential Moving Average(EMA)")
    return ModelEmaV3(model, decay=cfg.model.model_ema_decay, warmup=cfg.model.model_ema_warmup)


def setup_ddp_model(cfg: ExperimentConfig, model: nn.Module) -> DistributedDataParallel:
    if cfg.model.use_sync_bn:
        model = convert_sync_batchnorm(model)

    return DistributedDataParallel(
        model,
        device_ids=[get_local_rank()],
        output_device=get_local_rank(),
        find_unused_parameters=cfg.model.find_unused_parameters,
    )


def setup_fsdp_model(cfg: ExperimentConfig, model: nn.Module) -> FullyShardedDataParallel:
    amp_policy = None
    if cfg.use_amp:
        amp_policy = MixedPrecision(
            param_dtype=TORCH_DTYPE[cfg.amp_dtype],
            reduce_dtype=TORCH_DTYPE[cfg.amp_dtype],
            buffer_dtype=TORCH_DTYPE[cfg.amp_dtype],
        )
    cpu_offload = None
    if cfg.gpu.fsdp.use_cpu_offload:
        cpu_offload = CPUOffload(offload_params=True)
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy, min_num_params=cfg.gpu.fsdp.min_num_params
    )

    return FullyShardedDataParallel(
        model,
        device_id=get_local_rank(),
        cpu_offload=cpu_offload,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=amp_policy,
    )
