"""
CIFAR-10 data loading.

Images are kept in the range [0, 1]. Normalisation (subtracting the mean and
dividing by the standard deviation) happens INSIDE the model instead. This
means attack sizes such as eps = 8/255 always mean a real change in pixel
values, for every model (this avoids the epsilon-scaling problem of having
different input ranges for different models).

Use fake=True for a quick test without downloading CIFAR-10.
"""

import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _seed_worker(worker_id):
    """Gives each data-loading worker a fixed seed (for reproducibility)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _datasets(data_dir, fake, train_transform):
    test_transform = T.ToTensor()
    if fake:
        train = torchvision.datasets.FakeData(512, (3, 32, 32), 10, transform=train_transform)
        test = torchvision.datasets.FakeData(256, (3, 32, 32), 10, transform=test_transform,
                                             random_offset=1)
    else:
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True,
                                             transform=train_transform)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=test_transform)
    return train, test


def get_loaders(data_dir="./data", batch_size=128, num_workers=2, seed=0, fake=False):
    """Training loader (with standard augmentation) and test loader."""
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),   # shift the image by up to 4 pixels
        T.RandomHorizontalFlip(),      # mirror left-right half of the time
        T.ToTensor(),                  # pixels to [0, 1]
    ])
    train, test = _datasets(data_dir, fake, train_transform)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=_seed_worker,
                              generator=g, drop_last=True)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_calibration_loader(data_dir="./data", n=1000, seed=0, fake=False):
    """A small random set of TRAINING images, without augmentation, for PTQ calibration."""
    train, _ = _datasets(data_dir, fake, T.ToTensor())
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(len(train), generator=g)[:n].tolist()
    return DataLoader(Subset(train, idx), batch_size=100, shuffle=False)


def get_test_subset(data_dir="./data", n=1000, seed=0, fake=False):
    """A fixed random subset of the TEST set as tensors (used for attacks)."""
    _, test = _datasets(data_dir, fake, T.ToTensor())
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(len(test), generator=g)[:n].tolist()
    xs, ys = zip(*(test[i] for i in idx))
    return torch.stack(xs), torch.tensor(ys)
