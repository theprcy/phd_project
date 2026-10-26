"""
ResNet-20 for CIFAR-10 (He et al., 2016): about 0.27M parameters.

Written to be "quantisation-ready" for PyTorch:
  - QuantStub / DeQuantStub mark where the model switches to and from INT8.
  - The residual addition uses FloatFunctional, which can be quantised.
  - Each ReLU is its own module, so conv + batchnorm + relu can be fused.
"""

import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules, fuse_modules_qat

from src.models.common import Normalize


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()
        if stride != 1 or in_ch != out_ch:
            # 1x1 convolution so the shortcut matches the new shape
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.add = FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add.add(out, self.shortcut(x))
        return self.relu2(out)

    def fuse(self, qat=False):
        fuse = fuse_modules_qat if qat else fuse_modules
        fuse(self, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
        if isinstance(self.shortcut, nn.Sequential):
            fuse(self.shortcut, [["0", "1"]], inplace=True)


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.normalize = Normalize()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(16, 16, stride=1)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.dequant = DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _make_layer(in_ch, out_ch, stride, n_blocks=3):
        blocks = [BasicBlock(in_ch, out_ch, stride)]
        blocks += [BasicBlock(out_ch, out_ch, 1) for _ in range(n_blocks - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.normalize(x)          # stays in float
        x = self.quant(x)              # float -> INT8 (after conversion)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return self.dequant(x)         # INT8 -> float

    def fuse_model(self, qat=False):
        """Merge conv + batchnorm (+ relu) into single layers before quantisation."""
        fuse = fuse_modules_qat if qat else fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for layer in (self.layer1, self.layer2, self.layer3):
            for block in layer:
                block.fuse(qat)
