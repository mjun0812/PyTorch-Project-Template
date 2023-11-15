import math

import torch.nn as nn
import torch.nn.functional as F


class MaxPool2dSame(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        if isinstance(self.dilation, int):
            self.dilation = [self.dilation] * 2

        self.pool = nn.MaxPool2d(
            self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, x):
        h, w = x.shape[-2:]

        pad_h = (
            (math.ceil(h / self.stride[0]) - 1) * self.stride[0]
            + (self.kernel_size[0] - 1) * self.dilation[0]
            + 1
            - h
        )
        pad_h = max(pad_h, 0)
        pad_w = (
            (math.ceil(w / self.stride[1]) - 1) * self.stride[1]
            + (self.kernel_size[1] - 1) * self.dilation[1]
            + 1
            - w
        )
        pad_w = max(pad_w, 0)

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        x = F.pad(x, [left, right, top, bottom], value=0.0)

        return self.pool(x)
