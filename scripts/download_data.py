"""
Download CIFAR-10 once. Run this on the Hyperion LOGIN node (compute nodes have no
internet), with the proxy set:

    export http_proxy=http://hpc-proxy00.city.ac.uk:3128
    export https_proxy=http://hpc-proxy00.city.ac.uk:3128
    python scripts/download_data.py --data /full/path/to/data
"""

import argparse

import torchvision

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="folder to store CIFAR-10 in")
args = p.parse_args()

torchvision.datasets.CIFAR10(args.data, train=True, download=True)
torchvision.datasets.CIFAR10(args.data, train=False, download=True)
print(f"CIFAR-10 is ready in {args.data}")