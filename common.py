import torch
import torch.nn as nn

from src.data import CIFAR10_MEAN, CIFAR10_STD


class Normalize(nn.Module):
    """Normalises [0, 1] images inside the model (see src/data.py for why)."""

    def __init__(self, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std
