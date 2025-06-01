import math

import torch.nn.functional as F
from torch import nn


class MaxPool2dSame(nn.Module):
    """MaxPool2dSameは、"SAME"パディングを使用するMaxPooling層です。
    この層は、入力の空間サイズに基づいて適切なパディングを自動的に計算します。
    ただし、出力サイズは入力サイズとは異なる場合があります：
    stride=1の場合: 出力サイズは入力サイズと同じになります
    stride>1の場合: 出力サイズは入力サイズをstrideで割った値（切り上げ）に近くなります

    Args:
        kernel_size: カーネルサイズ。整数または2要素のタプル/リスト
        stride: ストライド。デフォルトはNone（kernel_sizeと同じ値）
        padding: 追加のパディング。デフォルトは0
        dilation: カーネル要素間の間隔。デフォルトは1
        return_indices: 最大値のインデックスを返すかどうか。デフォルトはFalse
        ceil_mode: 出力サイズの計算に切り上げを使用するかどうか。デフォルトはFalse

    Examples:
        # stride=1の場合、出力サイズは入力サイズと同じ
        >>> m = MaxPool2dSame(kernel_size=3, stride=1)
        >>> input = torch.randn(1, 3, 32, 32)
        >>> output = m(input)
        >>> print(output.shape)
        torch.Size([1, 3, 32, 32])

        # stride=2の場合、出力サイズは約半分になる
        >>> m = MaxPool2dSame(kernel_size=3, stride=2)
        >>> input = torch.randn(1, 3, 32, 32)
        >>> output = m(input)
        >>> print(output.shape)
        torch.Size([1, 3, 16, 16])
    """

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
