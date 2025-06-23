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
"""Registry for model classes."""


def build_model(cfg: ModelConfig, phase: PhaseStr = "train") -> BaseModel:
    """Build a model instance from configuration.

    Args:
        cfg: Model configuration containing class name, arguments, and weights.
        phase: Training phase for the model.

    Returns:
        Instantiated model with optional pre-trained weights loaded.
    """
    model = MODEL_REGISTRY.get(cfg.class_name)(cfg.args, cfg.loss, phase)
    if cfg.pre_trained_weight:
        load_model_weight(cfg.pre_trained_weight, model)
    return model


def calc_model_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Calculate the number of parameters in different model components.

    Args:
        model: PyTorch model to analyze.

    Returns:
        Tuple containing (total_params, trainable_params, backbone_params).
    """
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
    """Log model parameter counts to the logger.

    Args:
        model: PyTorch model to analyze and log.
    """
    num_params, num_trainable_params, num_backbone_params = calc_model_parameters(model)
    logger.info(f"Num Model Parameters: {num_params}")
    logger.info(f"Num Trainable Model Parameters: {num_trainable_params}")
    if num_backbone_params > 0:
        logger.info(f"Num Backbone Model Parameters: {num_backbone_params}")


def create_model_ema(cfg: ExperimentConfig, model: nn.Module) -> ModelEmaV3:
    """Create Exponential Moving Average wrapper for the model.

    Args:
        cfg: Experiment configuration containing EMA settings.
        model: PyTorch model to wrap with EMA.

    Returns:
        ModelEMA instance wrapping the input model.
    """
    logger.info("Use Model Exponential Moving Average(EMA)")
    return ModelEmaV3(model, decay=cfg.model.model_ema_decay, warmup=cfg.model.model_ema_warmup)


def setup_ddp_model(cfg: ExperimentConfig, model: nn.Module) -> DistributedDataParallel:
    """Setup model for Distributed Data Parallel training.

    Args:
        cfg: Experiment configuration containing DDP settings.
        model: PyTorch model to wrap with DDP.

    Returns:
        Model wrapped with DistributedDataParallel.
    """
    if cfg.model.use_sync_bn:
        model = convert_sync_batchnorm(model)

    return DistributedDataParallel(
        model,
        device_ids=[get_local_rank()],
        output_device=get_local_rank(),
        find_unused_parameters=cfg.model.find_unused_parameters,
    )


def setup_fsdp_model(cfg: ExperimentConfig, model: nn.Module) -> FullyShardedDataParallel:
    """Setup model for Fully Sharded Data Parallel training.

    Args:
        cfg: Experiment configuration containing FSDP settings.
        model: PyTorch model to wrap with FSDP.

    Returns:
        Model wrapped with FullyShardedDataParallel.
    """
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
