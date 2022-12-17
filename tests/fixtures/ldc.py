from __future__ import annotations

'''Lightweight Dense CNN for Edge Detection
It has less than 1 Million parameters
'''
import torch
from torch import nn
import torch.nn.functional as F

from typing import Any


class CoFusion(nn.Module):

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, 32, kernel_size=3, stride=1, padding=1
        )  # before 64
        self.conv3 = nn.Conv2d(
            32, out_ch, kernel_size=3, stride=1, padding=1
        )  # before 64  instead of 32
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32)  # before 64

    def forward(self, x: torch.Tensor) -> Any:
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features: int, out_features: int) -> None:
        super().__init__()

        self.add_module(
            'conv1', nn.Conv2d(
                input_features, out_features,
                kernel_size=3, stride=1, padding=2, bias=True
            )
        )
        self.add_module('norm1', nn.BatchNorm2d(out_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(
            out_features, out_features,
            kernel_size=3, stride=1, bias=True
            )
        )
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = inputs
        new_features = super().forward(F.relu(x1))
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(
        self, num_layers: int, input_features: int, out_features: int
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module(f'denselayer{i+1}', layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features: int, up_scale: int) -> None:
        super().__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(
        self, in_features: int, up_scale: int
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx: int, up_scale: int) -> int:
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x: torch.Tensor) -> Any:
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int,
        stride: int, use_bs: bool = True
    ) -> None:
        super().__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(
        self, in_features: int,
        mid_features: int,
        out_features: int | None = None,
        stride: int = 1, use_act: bool = True
    ) -> None:
        super().__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)
        return x


class LDC(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self) -> None:
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2,)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(2, 32, 64)  # [128,256,100,100]
        self.dblock_4 = _DenseBlock(3, 64, 96)  # 128
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.side_2 = SingleConvBlock(32, 64, 2)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(32, 64, 2)
        self.pre_dense_3 = SingleConvBlock(32, 64, 1)
        self.pre_dense_4 = SingleConvBlock(64, 96, 1)  # 128

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(64, 2)
        self.up_block_4 = UpConvBlock(96, 3)  # 128
        self.block_cat = CoFusion(4, 4)  # cats fusion method

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        assert x.ndim == 4, x.shape
        # supose the image size is 352x352
        # Block 1
        block_1 = self.block_1(x)  # [8,16,176,176]
        block_1_side = self.side_1(block_1)  # 16 [8,32,88,88]

        # Block 2
        block_2 = self.block_2(block_1)  # 32 - [8,32,176,176]
        block_2_down = self.maxpool(block_2)  # [8,32,88,88]
        block_2_add = block_2_down + block_1_side  # [8,32,88,88]
        block_2_side = self.side_2(block_2_add)  # [8,64,44,44] block 3 R connection

        # Block 3
        block_3_pre_dense = self.pre_dense_3(
            block_2_down)  # [8,64,88,88] block 3 L connection
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])  # [8,64,88,88]
        block_3_down = self.maxpool(block_3)  # [8,64,44,44]
        block_3_add = block_3_down + block_2_side  # [8,64,44,44]

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)  # [8,64,44,44]
        block_4_pre_dense = self.pre_dense_4(
            block_3_down+block_2_resize_half)  # [8,96,44,44]
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])  # [8,96,44,44]

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        results: list[torch.Tensor] = [out_1, out_2, out_3, out_4]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        results.append(block_cat)
        return results
