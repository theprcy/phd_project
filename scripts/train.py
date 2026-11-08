"""
Train an FP32 model on CIFAR-10 and save the weights and results.

Example:
    python -u scripts/train.py --model resnet20 --seed 0 --data /path/to/data --out /path/to/results
Quick test (no CIFAR-10 needed, 1 epoch on fake data):
    python -u scripts/train.py --model resnet20 --fake --epochs 1 --out ./test_results

Output in <out>/<model>_seed<seed>/:
    fp32.pt              final trained weights
    fp32_metrics.json    accuracy, size, latency, training time, run information
    history.csv          loss and accuracy for every epoch (for plotting)
    checkpoint.pt        saved progress, so a stopped job can continue

If checkpoint.pt exists, training continues from it automatically (use --no-resume to
start again). A resumed run is very close to, but not exactly the same as, an
uninterrupted run, because the random data order restarts.
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
from src.utils import (count_params, evaluate, get_device, measure_latency_ms, run_info,
                       save_csv, save_json, set_seed, state_dict_size_mb)


def train_one_epoch(model, loader, optimiser, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimiser.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return total_loss / total, 100.0 * correct / total


def save_checkpoint(path, epoch, model, optimiser, scheduler, history, minutes):
    """Save everything needed to continue training from this epoch."""
    tmp = path + ".tmp"
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimiser": optimiser.state_dict(), "scheduler": scheduler.state_dict(),
                "history": history, "train_minutes": minutes}, tmp)
    os.replace(tmp, path)  # write to a temporary file first, so a crash never leaves a broken file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet20"], required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./results")
    p.add_argument("--eval-every", type=int, default=1, help="test accuracy every N epochs")
    p.add_argument("--ckpt-every", type=int, default=5, help="save progress every N epochs")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint")
    p.add_argument("--fake", action="store_true", help="fake data for a quick test")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    run_dir = os.path.join(args.out, f"{args.model}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    history_path = os.path.join(run_dir, "history.csv")
    print(f"Training {args.model} (seed {args.seed}) on {device}", flush=True)
    print(f"Results folder: {os.path.abspath(run_dir)}", flush=True)

    train_loader, test_loader = get_loaders(args.data, args.batch_size, seed=args.seed,
                                            fake=args.fake)
    model = build_model(args.model).to(device)

    # Standard CIFAR settings: SGD with momentum, learning rate falls along a cosine curve
    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    # Continue from a checkpoint if one exists
    start_epoch, history, minutes_before = 1, [], 0.0
    if os.path.exists(ckpt_path) and not args.no_resume:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        minutes_before = ckpt["train_minutes"]
        print(f"Resuming from epoch {start_epoch}", flush=True)

    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        lr = optimiser.param_groups[0]["lr"]  # learning rate used in this epoch
        loss, train_acc = train_one_epoch(model, train_loader, optimiser, device)
        scheduler.step()

        test_acc = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_acc = evaluate(model, test_loader, device)

        history.append({"epoch": epoch, "lr": round(lr, 6), "train_loss": round(loss, 4),
                        "train_acc": round(train_acc, 2),
                        "test_acc": None if test_acc is None else round(test_acc, 2)})
        save_csv(history, history_path)

        msg = f"epoch {epoch:3d}  lr {lr:.4f}  loss {loss:.3f}  train {train_acc:.2f}%"
        if test_acc is not None:
            msg += f"  test {test_acc:.2f}%"
        print(msg, flush=True)

        minutes = minutes_before + (time.time() - start) / 60
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            save_checkpoint(ckpt_path, epoch, model, optimiser, scheduler, history, minutes)

    # Final evaluation and saving
    test_acc = evaluate(model, test_loader, device)
    ckpt_file = os.path.join(run_dir, "fp32.pt")
    torch.save(model.state_dict(), ckpt_file)
    print(f"Saved {ckpt_file}", flush=True)

    model.cpu()
    save_json({
        "model": args.model,
        "precision": "fp32",
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "test_acc": test_acc,
        "params": count_params(model),
        "size_mb": state_dict_size_mb(model),
        "cpu_latency_ms": measure_latency_ms(model),
        "train_time_min": minutes_before + (time.time() - start) / 60,
        "run_info": run_info(),
    }, os.path.join(run_dir, "fp32_metrics.json"))


if __name__ == "__main__":
    main()