"""
MobileNetV2 for 32x32 CIFAR-10 images with 2,237,770 parameters.

The architecture follows chenhang98/mobileNet-v2_cifar10
(https://github.com/chenhang98/mobileNet-v2_cifar10, network.py and blocks.py),
which adapts MobileNetV2 (Sandler et al., 2018) to CIFAR-10.

Differences from the ImageNet version in the paper (same as the reference repo):
  - The first convolution and the 2nd and 3rd stages use stride 1 instead of 2.
    ImageNet images are 224x224 and the original shrinks them 32 times (224 -> 7).
    CIFAR images are only 32x32, so here they shrink 4 times (32 -> 8).
  - The first block (expand ratio 1) still has its 1x1 expand convolution.
    The paper and torchvision leave it out when the expand ratio is 1.

Differences from the reference repo (report these in the methods section):
  - ReLU instead of ReLU6, because PyTorch's quantisation can only fuse
    conv + batchnorm + ReLU (torchvision's quantisable MobileNetV2 does the same).
  - Normalisation happens inside the model (see src/data.py for why).
  - Written for PyTorch eager-mode quantisation: QuantStub/DeQuantStub, a
    FloatFunctional for the addition, and fuse functions (same as ResNet-20).
"""

import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules, fuse_modules_qat

from src.models.common import Normalize


def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, groups=1):
    """Convolution -> BatchNorm -> ReLU, kept together so they can be fused later.
    groups=in_ch makes a depthwise convolution: each channel gets its own filter,
    instead of every filter looking at all channels."""
    return nn.Sequential(
        # padding = kernel // 2 keeps the size the same when stride is 1
        # (padding 1 for a 3x3 kernel, padding 0 for a 1x1 kernel)
        nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


class InvertedResidual(nn.Module):
    """The MobileNetV2 block (BaseBlock in the reference repo):
    expand (1x1) -> depthwise (3x3) -> project (1x1).

    1. Expand: a 1x1 convolution increases the channels by expand_ratio (usually 6).
    2. Depthwise: a 3x3 convolution on each channel separately. This is much cheaper
       than a normal 3x3 convolution, and is why MobileNetV2 is efficient.
    3. Project: a 1x1 convolution reduces the channels to out_ch. There is no ReLU
       after this step (the paper calls it a linear bottleneck), because ReLU on a
       small number of channels throws away too much information.
    A shortcut adds the input to the output when they have the same shape.
    """

    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio  # number of channels inside the block
        # The input can only be added if the size and channels don't change
        self.use_shortcut = stride == 1 and in_ch == out_ch
        self.conv = nn.Sequential(
            # expand (kept even when expand_ratio is 1, as in the reference repo)
            conv_bn_relu(in_ch, hidden, kernel=1),
            # depthwise
            conv_bn_relu(hidden, hidden, kernel=3, stride=stride, groups=hidden),
            # project (no ReLU)
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # FloatFunctional instead of "+", so the addition can be quantised
        self.add = FloatFunctional()

    def forward(self, x):
        if self.use_shortcut:
            return self.add.add(x, self.conv(x))
        return self.conv(x)

    def fuse(self, qat=False):
        """Merge conv + batchnorm (+ relu) inside this block before quantisation."""
        fuse = fuse_modules_qat if qat else fuse_modules
        # Positions inside self.conv: 0 = expand group, 1 = depthwise group,
        # 2 = project conv, 3 = project BN
        fuse(self.conv[0], [["0", "1", "2"]], inplace=True)  # expand: conv + BN + ReLU
        fuse(self.conv[1], [["0", "1", "2"]], inplace=True)  # depthwise: conv + BN + ReLU
        fuse(self.conv, [["2", "3"]], inplace=True)          # project: conv + BN


class MobileNetV2(nn.Module):
    # Each row is one stage, as in Table 2 of the paper:
    # (expand ratio t, output channels c, number of blocks n, stride of the first block s)
    CONFIG = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),   # stride 1 for CIFAR (2 in the paper)
        (6, 32, 3, 1),   # stride 1 for CIFAR (2 in the paper)
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    # Image size through the network: 32x32 until the 4th stage, then 16x16,
    # and 8x8 after the 6th stage

    def __init__(self, num_classes=10):
        super().__init__()
        self.normalize = Normalize()   # [0, 1] pixels -> normalised values
        self.quant = QuantStub()       # marks where the quantised part starts
        layers = [conv_bn_relu(3, 32, stride=1)]  # stride 1 for CIFAR (2 in the paper)
        in_ch = 32
        for t, c, n, s in self.CONFIG:
            for i in range(n):
                # Only the first block of a stage uses stride s; the rest use stride 1
                layers.append(InvertedResidual(in_ch, c, s if i == 0 else 1, t))
                in_ch = c
        layers.append(conv_bn_relu(in_ch, 1280, kernel=1))  # last 1x1 conv: 320 -> 1280 channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)     # average each channel -> 1280 values
        self.fc = nn.Linear(1280, num_classes)  # one score per class
        self.dequant = DeQuantStub()   # marks where the quantised part ends

        # Same initialisation as the reference repo: convolution weights from a normal
        # distribution with spread sqrt(2 / (kernel height x kernel width x output
        # channels)), which is Kaiming normal with mode="fan_out". BatchNorm starts at
        # weight 1 and bias 0 (PyTorch's default). The Linear layer keeps PyTorch's default.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x):
        x = self.normalize(x)          # stays in float
        x = self.quant(x)              # float -> INT8 (after conversion)
        x = self.features(x)
        x = self.pool(x).flatten(1)    # (batch, 1280, 1, 1) -> (batch, 1280)
        x = self.fc(x)
        return self.dequant(x)         # INT8 -> float

    def fuse_model(self, qat=False):
        """Merge conv + batchnorm (+ relu) into single layers before quantisation."""
        fuse = fuse_modules_qat if qat else fuse_modules
        for m in self.features:
            if isinstance(m, InvertedResidual):
                m.fuse(qat)
            else:  # the conv_bn_relu groups at the start and the end
                fuse(m, [["0", "1", "2"]], inplace=True)