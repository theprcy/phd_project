"""
Train an FP32 model on CIFAR-10 and save the weights and results.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_loaders
from src.models import build_model
from src.utils import (count_params, evaluate, get_device, measure_latency_ms, save_json,
                       set_seed, state_dict_size_mb)


def train_one_epoch(model, loader, optimiser, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return total_loss / total, 100.0 * correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet20", "mobilenetv2"], required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./results")
    p.add_argument("--fake", action="store_true", help="fake data for a quick test")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    run_dir = os.path.join(args.out, f"{args.model}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Training {args.model} (seed {args.seed}) on {device}")

    train_loader, test_loader = get_loaders(args.data, args.batch_size, seed=args.seed,
                                            fake=args.fake)
    model = build_model(args.model).to(device)

    # Standard CIFAR settings: SGD with momentum, learning rate falls along a cosine curve
    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, optimiser, device)
        scheduler.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            test_acc = evaluate(model, test_loader, device)
            print(f"epoch {epoch:3d}  loss {loss:.3f}  train {train_acc:.2f}%  test {test_acc:.2f}%")

    test_acc = evaluate(model, test_loader, device)
    ckpt = os.path.join(run_dir, "fp32.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Saved {ckpt}")

    model.cpu()
    save_json({
        "model": args.model,
        "precision": "fp32",
        "seed": args.seed,
        "epochs": args.epochs,
        "test_acc": test_acc,
        "params": count_params(model),
        "size_mb": state_dict_size_mb(model),
        "cpu_latency_ms": measure_latency_ms(model),
        "train_time_min": (time.time() - start) / 60,
    }, os.path.join(run_dir, "fp32_metrics.json"))


if __name__ == "__main__":
    main()
