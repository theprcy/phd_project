"""
Train an FP32 model on CIFAR-10 and save the weights and results.
"""

import argparse
import os
import sys
import time # to measure the training time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_loaders
from src.models import build_model
from src.utils import (count_params, evaluate, get_device, measure_latency_ms, run_info,
                       save_csv, save_json, set_seed, state_dict_size_mb)


def train_one_epoch(model, loader, optimiser, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss() # using cross entropy loss which is a standard loss for classification
    total_loss, correct, total = 0.0, 0, 0 
    for x, y in loader:  # x = a batch of images, y = their correct labels
        # Move the batch to the GPU. non_blocking lets the copy run in the
        # background while the GPU does other work.
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # training steps
        optimiser.zero_grad() # 1) clear gradients from previous batch
        logits = model(x) # 2) forward pass
        loss = loss_fn(logits, y) # 3) calculate the loss
        loss.backward() # 4) see how much weight to change
        optimiser.step() # 5) change/update the weight
        total_loss += loss.item() * y.numel() # the loss is the avg over the batch
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return total_loss / total, 100.0 * correct / total

# Settings that must be the same when a run continues from a checkpoint
RESUME_KEYS = ["model", "epochs", "lr", "batch_size", "weight_decay", "seed"]

def save_checkpoint(path, epoch, model, optimiser, scheduler, history, minutes, args):
    """Save everything needed to continue training from this epoch"""
    tmp = path + ".tmp"
    torch.save({"epoch": epoch,                         # last finished epoch
                "model": model.state_dict(),            # the weights
                "optimiser": optimiser.state_dict(),    # includes momentum, needed to continue smoothly
                "scheduler": scheduler.state_dict(),    # where we are in the learning rate schedule
                "history": history,                     # per-epoch results so far
                "train_minutes": minutes,               # training time so far
                "args": vars(args)},                    # the settings, checked when resuming  
               tmp)
    os.replace(tmp, path)  # write to a temporary file first, so a crash never leaves a broken file

def check_checkpoint_settings(ckpt, args):
    """Stop if the checkpoint was made with different settings from this run"""
    problems = []
    saved = ckpt.get("args")  # None if the checkpoint was made by older code
    if saved is None:
        problems.append("the checkpoint has no saved settings (made by older code)")
    else:
        for key in RESUME_KEYS:
            if saved.get(key) != getattr(args, key):  # getattr(args, "lr") is the same as args.lr
                problems.append(f"{key}: checkpoint {saved.get(key)}, this run {getattr(args, key)}")
    # The scheduler's length must match the number of epochs (this caught the seed 0 problem)
    t_max = ckpt["scheduler"].get("T_max")
    if t_max != args.epochs:
        problems.append(f"scheduler length: checkpoint {t_max}, this run {args.epochs}")
    if problems:
        # sys.exit stops the program and prints the message
        sys.exit("Cannot continue from checkpoint:\n  " + "\n  ".join(problems) +
                 "\nUse --no-resume to start again, or a different --out folder.")

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

    ## Set up seeds, device and output folder
    set_seed(args.seed)
    device = get_device()
    run_dir = os.path.join(args.out, f"{args.model}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True) # set up outputs folder
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    history_path = os.path.join(run_dir, "history.csv")
    print(f"Training {args.model} (seed {args.seed}) on {device}", flush=True)
    print(f"Results folder: {os.path.abspath(run_dir)}", flush=True)
    # flush=True prints straight away, so the Slurm log file updates during the run

    # Record settings and code version now, so stopped runs also have a record.
    # One file per Slurm job, so a resumed run shows up as a second file.
    job = os.environ.get("SLURM_JOB_ID", "local")  # "local" when not running under Slurm
    save_json({"args": vars(args), "run_info": run_info()},
              os.path.join(run_dir, f"run_info_{job}.json"))

    # train model
    train_loader, test_loader = get_loaders(args.data, args.batch_size, seed=args.seed,
                                            fake=args.fake)
    model = build_model(args.model).to(device)

    ## Optimiser and scheduler: SGD with momentum, learning rate falls along a cosine curve
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
        "cpu_latency_threads": 1,
        "train_time_min": minutes_before + (time.time() - start) / 60,
        "run_info": run_info(),
    }, os.path.join(run_dir, "fp32_metrics.json"))


if __name__ == "__main__":
    main()