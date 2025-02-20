from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ActivationFnTypes = Literal[
    "ReLU", "GELU", "GLU", "PReLU", "SELU", "Swish", "MemoryEfficientSwish", "SiLU"
]
ActivationLayerTypes = ActivationFnTypes


def get_activation_fn(activation: ActivationFnTypes) -> nn.Module:
    """Return an activation function given a string"""
    if activation == "ReLU":
        return F.relu
    elif activation == "GELU":
        return F.gelu
    elif activation == "GLU":
        return F.glu
    elif activation == "PReLU":
        return F.prelu
    elif activation == "SELU":
        return F.selu
    elif activation == "Swish":
        return swish_fn
    elif activation == "MemoryEfficientSwish":
        return memory_efficient_swish_fn
    elif activation == "SiLU":
        return F.silu
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_activation_layer(activation: ActivationLayerTypes) -> nn.Module:
    if activation == "ReLU":
        return nn.ReLU
    elif activation == "GELU":
        return nn.GELU
    elif activation == "GLU":
        return nn.GLU
    elif activation == "PReLU":
        return nn.PReLU
    elif activation == "SELU":
        return nn.SELU
    elif activation == "Swish":
        return Swish
    elif activation == "MemoryEfficientSwish":
        return MemoryEfficientSwish
    elif activation == "SiLU":
        return nn.SiLU
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def swish_fn(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return swish_fn(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


def memory_efficient_swish_fn(x: torch.Tensor) -> torch.Tensor:
    return SwishImplementation.apply(x)


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return memory_efficient_swish_fn(x)


ActivationLayer = [
    nn.ReLU,
    nn.GELU,
    nn.GLU,
    nn.PReLU,
    nn.SELU,
    Swish,
    MemoryEfficientSwish,
    nn.SiLU,
]
