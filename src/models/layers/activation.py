from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from ...utils import filter_kwargs

ActivationNames = Literal[
    "ReLU", "GELU", "GLU", "PReLU", "SELU", "Swish", "MemoryEfficientSwish", "SiLU"
]


def swish_fn(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function.

    Args:
        x: Input tensor.

    Returns:
        Activated tensor using Swish function.
    """
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    """Swish activation layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation.

        Args:
            x: Input tensor.

        Returns:
            Swish-activated tensor.
        """
        return swish_fn(x)


class SwishImplementation(torch.autograd.Function):
    """Memory-efficient Swish implementation with custom gradient."""

    @staticmethod
    def forward(ctx: Any, i: torch.Tensor) -> torch.Tensor:
        """Forward pass of Swish activation.

        Args:
            ctx: Context object to save tensors for backward pass.
            i: Input tensor.

        Returns:
            Swish-activated tensor.
        """
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass of Swish activation.

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Gradient of the loss with respect to output.

        Returns:
            Gradient of the loss with respect to input.
        """
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


def memory_efficient_swish_fn(x: torch.Tensor) -> torch.Tensor:
    """Memory-efficient Swish activation function.

    Args:
        x: Input tensor.

    Returns:
        Swish-activated tensor with custom gradient computation.
    """
    return SwishImplementation.apply(x)


class MemoryEfficientSwish(nn.Module):
    """Memory-efficient Swish activation layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply memory-efficient Swish activation.

        Args:
            x: Input tensor.

        Returns:
            Swish-activated tensor.
        """
        return memory_efficient_swish_fn(x)


def get_activation_fn(activation: ActivationNames) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get activation function by name.

    Args:
        activation: Name of the activation function.

    Returns:
        Activation function callable.
    """
    activation_fns = {
        "ReLU": F.relu,
        "GELU": F.gelu,
        "GLU": F.glu,
        "PReLU": F.prelu,
        "SELU": F.selu,
        "Swish": swish_fn,
        "MemoryEfficientSwish": memory_efficient_swish_fn,
        "SiLU": F.silu,
    }
    return activation_fns[activation]


def get_activation_layer(activation: ActivationNames) -> type[nn.Module]:
    """Get activation layer class by name.

    Args:
        activation: Name of the activation layer.

    Returns:
        Activation layer class.
    """
    activation_classes = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "GLU": nn.GLU,
        "PReLU": nn.PReLU,
        "SELU": nn.SELU,
        "Swish": Swish,
        "MemoryEfficientSwish": MemoryEfficientSwish,
        "SiLU": nn.SiLU,
    }
    return activation_classes[activation]


def build_activation_layer(activation: ActivationNames, **kwargs: Any) -> nn.Module:
    """Build activation layer with filtered kwargs.

    Args:
        activation: Name of the activation layer.
        **kwargs: Additional arguments for the layer constructor.

    Returns:
        Initialized activation layer.
    """
    cls = get_activation_layer(activation)
    filtered_kwargs = filter_kwargs(cls, kwargs)
    return cls(**filtered_kwargs)
