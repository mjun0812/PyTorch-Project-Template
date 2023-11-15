import math

import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Conv2dSame(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True
    ):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2

        if isinstance(self.dilation, int):
            self.dilation = [self.dilation] * 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
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
        return self.conv(x)
