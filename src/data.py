"""
CIFAR-10 data loading.

Images are kept in the range [0, 1]. Normalisation (subtracting the mean and
dividing by the standard deviation) happens inside the model instead. This
means attack sizes such as eps = 8/255 always mean a real change in pixel
values, for every model (this avoids the epsilon-scaling problem of having
different input ranges for different models)

Use fake=True for a quick test without downloading CIFAR-10
"""

import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

# Everything that differs between the datasets, looked up by name
DATASETS = {
    "cifar10": {"cls": torchvision.datasets.CIFAR10, "num_classes": 10,
                "mean": CIFAR10_MEAN, "std": CIFAR10_STD},
    "cifar100": {"cls": torchvision.datasets.CIFAR100, "num_classes": 100,
                 "mean": CIFAR100_MEAN, "std": CIFAR100_STD},
}


def default_num_workers():
    """Use the CPU cores Slurm gave this job (SLURM_CPUS_PER_TASK); 2 when not on the cluster"""
    return int(os.environ.get("SLURM_CPUS_PER_TASK", 2))


def _seed_worker(worker_id):
    """Gives each data-loading worker a fixed seed (for reproducibility)"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _datasets(data_dir, fake, train_transform, dataset="cifar10"):
    test_transform = T.ToTensor()
    info = DATASETS[dataset]
    if fake:
        train = torchvision.datasets.FakeData(512, (3, 32, 32), 10, transform=train_transform)
        test = torchvision.datasets.FakeData(256, (3, 32, 32), 10, transform=test_transform,
                                             random_offset=1)
    else:
        # download=True only downloads if the files are missing
        train = info["cls"](data_dir, train=True, download=True, transform=train_transform)
        test = info["cls"](data_dir, train=False, download=True, transform=test_transform)
    return train, test


def get_loaders(data_dir="./data", batch_size=128, num_workers=None, seed=0, fake=False,
                dataset="cifar10"):
    """Training loader (with standard augmentation) and test loader."""
    if num_workers is None:
        num_workers = default_num_workers()
    pin = torch.cuda.is_available()  # faster copying of images to the GPU

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),   # shift the image by up to 4 pixels
        T.RandomHorizontalFlip(),      # mirror left-right half of the time
        T.ToTensor(),                  # pixels to [0, 1]
    ])
    train, test = _datasets(data_dir, fake, train_transform, dataset)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=_seed_worker,
                              generator=g, drop_last=True, pin_memory=pin,
                              persistent_workers=num_workers > 0)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=num_workers,
                             pin_memory=pin, persistent_workers=num_workers > 0)
    return train_loader, test_loader