from typing import Any, Literal

import torch
from torch import nn

from ...utils import filter_kwargs

NormLayerNames = Literal[
    "BatchNorm2d", "GroupNorm", "InstanceNorm2d", "LayerNorm", "FrozenBatchNorm2d", "Identity"
]


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n: int) -> None:
        """Initialize FrozenBatchNorm2d layer.

        Args:
            n: Number of features/channels.
        """
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
        """Load parameters from state dict, removing num_batches_tracked key.

        Args:
            state_dict: State dictionary containing parameters.
            prefix: Parameter name prefix.
            local_metadata: Local metadata.
            strict: Whether to strictly enforce that the keys match.
            missing_keys: List to store missing parameter names.
            unexpected_keys: List to store unexpected parameter names.
            error_msgs: List to store error messages.
        """
        num_batches_tracked_key = prefix + "num_batches_tracked"
        state_dict.pop(num_batches_tracked_key, None)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frozen batch normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
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


def get_norm_layer(norm_type: NormLayerNames) -> nn.Module:
    """Get normalization layer class by name.

    Args:
        norm_type: Name of the normalization layer.

    Returns:
        Normalization layer class.
    """
    norm_classes = {
        "BatchNorm2d": nn.BatchNorm2d,
        "GroupNorm": nn.GroupNorm,
        "InstanceNorm2d": nn.InstanceNorm2d,
        "LayerNorm": nn.LayerNorm,
        "FrozenBatchNorm2d": FrozenBatchNorm2d,
        "Identity": nn.Identity,
    }
    return norm_classes[norm_type]


def build_norm_layer(norm_type: NormLayerNames, **kwargs: Any) -> nn.Module:
    """Build normalization layer with filtered kwargs.

    Args:
        norm_type: Name of the normalization layer.
        **kwargs: Additional arguments for the layer constructor.

    Returns:
        Initialized normalization layer.
    """
    cls = get_norm_layer(norm_type)
    filtered_kwargs = filter_kwargs(cls, kwargs)
    return cls(**filtered_kwargs)
