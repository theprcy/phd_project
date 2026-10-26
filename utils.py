"""Shared helpers used by all scripts."""

import json
import os
import random
import tempfile
import time

import numpy as np
import torch


def set_seed(seed):
    """Fix all random seeds so an experiment can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


def state_dict_size_mb(model):
    """Size of the saved weights in MB (what would be stored on the device)."""
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return file_size_mb(f.name)


@torch.no_grad()
def evaluate(model, loader, device):
    """Top-1 accuracy (%) on a data loader."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


@torch.no_grad()
def predict_probs(model, x, device, batch_size=100):
    """Softmax probabilities for a tensor of images, as a NumPy array."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size].to(device))
        out.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def measure_latency_ms(model, n_runs=100, n_warmup=10):
    """Average time (ms) for one 32x32 image on the CPU, batch size 1."""
    model.eval()
    x = torch.rand(1, 3, 32, 32)
    for _ in range(n_warmup):
        model(x)
    start = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    return 1000.0 * (time.perf_counter() - start) / n_runs
