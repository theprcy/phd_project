"""Shared helpers used by all scripts."""

import csv
import json
import os
import platform
import random
import subprocess
import tempfile
import time

import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    print(f"Saved {path}", flush=True)


def save_csv(rows, path):
    """Save a list of dictionaries (one per epoch) as a CSV file, e.g. for Google Sheets."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)  # missing values (None) are written as empty cells


def run_info():
    """Records how a result was produced: software, hardware, Slurm job and code version."""
    info = {
        "pytorch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        info["git_commit"] = None
    return info


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
def measure_latency_ms(model, n_runs=100, n_warmup=10):
    """Average time (ms) for one 32x32 image on the CPU, batch size 1.
    Only compare values measured on the same machine."""
    model.eval()
    x = torch.rand(1, 3, 32, 32)
    for _ in range(n_warmup):
        model(x)
    start = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    return 1000.0 * (time.perf_counter() - start) / n_runs