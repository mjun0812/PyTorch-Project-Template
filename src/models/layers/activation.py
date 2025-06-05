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
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish_fn(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, i: torch.Tensor) -> torch.Tensor:
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


def memory_efficient_swish_fn(x: torch.Tensor) -> torch.Tensor:
    return SwishImplementation.apply(x)


class MemoryEfficientSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return memory_efficient_swish_fn(x)


def get_activation_fn(activation: ActivationNames) -> Callable[[torch.Tensor], torch.Tensor]:
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
    cls = get_activation_layer(activation)
    filtered_kwargs = filter_kwargs(cls, kwargs)
    return cls(**filtered_kwargs)
