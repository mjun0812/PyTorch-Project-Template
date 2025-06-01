from torch import nn


class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution 2D.

    通常の畳み込みをdepthwise convolutionとpointwise convolutionに分解することで、
    パラメータ数と計算量を削減します。

    depthwise convolution: 各入力チャンネルに対して独立した空間的な畳み込みを適用
    pointwise convolution: 1x1の畳み込みを使用してチャンネル間の情報を混合

    Attributes:
        depthwise_conv: 空間方向の畳み込み層
        pointwise_conv: チャンネル方向の畳み込み層
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 空間方向の畳み込み
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        # チャンネル方向の畳み込み
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))
