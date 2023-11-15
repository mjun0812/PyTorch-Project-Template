import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    elif activation == "prelu":
        return nn.PReLU()
    elif activation == "selu":
        return F.selu
    elif activation == "swish":
        return Swish()
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


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
