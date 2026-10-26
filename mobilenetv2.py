"""
MobileNetV2 (Sandler et al., 2018) adapted for 32x32 CIFAR images: about 2.2M parameters.

Changes from the ImageNet version:
  - The first convolution and the second stage use stride 1 instead of 2,
    because CIFAR images are already small (32x32 instead of 224x224).
  - ReLU is used instead of ReLU6, because PyTorch can fuse conv + batchnorm + ReLU
    for quantisation (torchvision's quantisable MobileNetV2 does the same).
"""

import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules, fuse_modules_qat

from src.models.common import Normalize


def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


class InvertedResidual(nn.Module):
    """Expand (1x1) -> depthwise (3x3) -> project (1x1), with a shortcut when shapes match."""

    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_shortcut = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn_relu(in_ch, hidden, kernel=1))
        layers += [
            conv_bn_relu(hidden, hidden, kernel=3, stride=stride, groups=hidden),  # depthwise
            nn.Conv2d(hidden, out_ch, 1, bias=False),                              # project
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)
        self.add = FloatFunctional()

    def forward(self, x):
        if self.use_shortcut:
            return self.add.add(x, self.conv(x))
        return self.conv(x)

    def fuse(self, qat=False):
        fuse = fuse_modules_qat if qat else fuse_modules
        for m in self.conv:
            if isinstance(m, nn.Sequential):
                fuse(m, [["0", "1", "2"]], inplace=True)
        n = len(self.conv)
        fuse(self.conv, [[str(n - 2), str(n - 1)]], inplace=True)  # project conv + bn


class MobileNetV2(nn.Module):
    # (expand ratio t, output channels c, number of blocks n, first stride s)
    CONFIG = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),   # stride 1 for CIFAR (2 in the ImageNet version)
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10):
        super().__init__()
        self.normalize = Normalize()
        self.quant = QuantStub()
        layers = [conv_bn_relu(3, 32, stride=1)]  # stride 1 for CIFAR
        in_ch = 32
        for t, c, n, s in self.CONFIG:
            for i in range(n):
                layers.append(InvertedResidual(in_ch, c, s if i == 0 else 1, t))
                in_ch = c
        layers.append(conv_bn_relu(in_ch, 1280, kernel=1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)
        self.dequant = DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x):
        x = self.normalize(x)
        x = self.quant(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.fc(self.dropout(x))
        return self.dequant(x)

    def fuse_model(self, qat=False):
        fuse = fuse_modules_qat if qat else fuse_modules
        for m in self.features:
            if isinstance(m, InvertedResidual):
                m.fuse(qat)
            else:  # conv_bn_relu blocks at the start and end
                fuse(m, [["0", "1", "2"]], inplace=True)
