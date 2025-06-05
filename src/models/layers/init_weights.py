import math

import torch


def variance_scaling(w: torch.Tensor, gain: float = 1, groups: int = 1) -> None:
    fan_in, _ = fan_in_out(w, groups)
    gain /= max(1.0, fan_in)  # fan in
    # gain /= max(1., (fan_in + fan_out) / 2.)  # fan

    # should it be normal or trunc normal? using normal for now since no good trunc in PT
    # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
    # std = math.sqrt(gain) / .87962566103423978
    # w.data.trunc_normal(std=std)
    std = math.sqrt(gain)
    w.data.normal_(std=std)


def fan_in_out(w: torch.Tensor, groups: int = 1) -> tuple[int, int]:
    dimensions = w.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )
    num_input_fmaps = w.size(1)
    num_output_fmaps = w.size(0)
    receptive_field_size = 1
    if w.dim() > 2:
        receptive_field_size = w[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    fan_out //= groups
    return fan_in, fan_out


def glorot_uniform(w: torch.Tensor, gain: float = 1, groups: int = 1) -> None:
    fan_in, fan_out = fan_in_out(w, groups)
    gain /= max(1.0, (fan_in + fan_out) / 2.0)  # fan avg
    limit = math.sqrt(3.0 * gain)
    w.data.uniform_(-limit, limit)
