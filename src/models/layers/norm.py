from functools import partial
from typing import Any, Literal

import torch
from torch import nn

NormLayerTypes = Literal[
    "BatchNorm2d", "GroupNorm", "InstanceNorm2d", "LayerNorm", "FrozenBatchNorm2d", "Identity"
]


def get_norm_layer(norm_type: NormLayerTypes, **kwargs: Any) -> type[nn.Module]:
    if norm_type == "BatchNorm2d":
        return nn.BatchNorm2d
    elif norm_type == "GroupNorm":
        return partial(nn.GroupNorm, **kwargs)
    elif norm_type == "InstanceNorm2d":
        return nn.InstanceNorm2d
    elif norm_type == "LayerNorm":
        return nn.LayerNorm
    elif norm_type == "FrozenBatchNorm2d":
        return FrozenBatchNorm2d
    elif norm_type == "Identity":
        return nn.Identity


def build_norm_layer(input_dim: int, norm_type: NormLayerTypes, **kwargs: Any) -> nn.Module:
    if norm_type == "BatchNorm2d":
        return nn.BatchNorm2d(input_dim, **kwargs)
    elif norm_type == "GroupNorm":
        return nn.GroupNorm(num_channels=input_dim, **kwargs)
    elif norm_type == "InstanceNorm2d":
        return nn.InstanceNorm2d(input_dim, **kwargs)
    elif norm_type == "LayerNorm":
        return nn.LayerNorm(input_dim, **kwargs)
    elif norm_type == "FrozenBatchNorm2d":
        return FrozenBatchNorm2d(input_dim, **kwargs)
    elif norm_type == "Identity" or norm_type is None:
        return nn.Identity()


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        num_batches_tracked_key = prefix + "num_batches_tracked"
        state_dict.pop(num_batches_tracked_key, None)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


NormLayers = [
    nn.BatchNorm2d,
    nn.GroupNorm,
    nn.InstanceNorm2d,
    nn.LayerNorm,
    FrozenBatchNorm2d,
    nn.Identity,
]
