"""
Train the simulated low-bit versions: QAT 4-bit, TWN (ternary) and SQ-TWN (stochastic ternary).

By default each version starts from the trained FP32 model and is fine-tuned, with the
SAME number of epochs for every version, so the comparison is fair.
For SQ-TWN, those epochs are split into four stages: 50%, 75%, 87.5%, 100% of filters quantised.

Examples:
    python scripts/lowbit.py --model resnet20 --version qat4  --seed 0
    python scripts/lowbit.py --model resnet20 --version twn   --seed 0
    python scripts/lowbit.py --model resnet20 --version sqtwn --seed 0
Follow the SQ paper more closely (train from scratch, longer):
    python scripts/lowbit.py --model resnet20 --version sqtwn --from-scratch --epochs 200 --lr 0.1

Output in results/<model>_seed<seed>/: <version>.pt and <version>_metrics.json
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import train_one_epoch
from src.data import get_loaders
from src.lowbit import LOWBIT_VERSIONS, build_lowbit_model, set_sq_ratio, theoretical_size_mb
from src.utils import count_params, evaluate, get_device, save_json, set_seed

SQ_STAGES = [0.5, 0.75, 0.875, 1.0]  # Dong et al. (2017), "exponential" scheme


def train_epochs(model, train_loader, test_loader, device, epochs, lr, label):
    """Train for some epochs with SGD and a cosine learning-rate schedule."""
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    for epoch in range(1, epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, optimiser, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            test_acc = evaluate(model, test_loader, device)
            print(f"{label} epoch {epoch:3d}  loss {loss:.3f}  train {train_acc:.2f}%  "
                  f"test {test_acc:.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet20", "mobilenetv2"], required=True)
    p.add_argument("--version", choices=list(LOWBIT_VERSIONS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40, help="total epochs (split into 4 for SQ)")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--from-scratch", action="store_true", help="do not start from the FP32 model")
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./results")
    p.add_argument("--fake", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    run_dir = os.path.join(args.out, f"{args.model}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. Build the model, load FP32 weights (unless from scratch), then swap in low-bit layers
    model = build_lowbit_model(args.model, args.version)
    if not args.from_scratch:
        fp32 = torch.load(os.path.join(run_dir, "fp32.pt"), map_location="cpu")
        missing, unexpected = model.load_state_dict(fp32, strict=False)
        # the only new entries are the activation-range buffers of the low-bit layers
        assert not unexpected and all("act_quant" in k for k in missing), (missing, unexpected)
    model = model.to(device)

    train_loader, test_loader = get_loaders(args.data, args.batch_size, seed=args.seed,
                                            fake=args.fake)

    # 2. Train
    if args.version == "sqtwn":
        per_stage = max(1, args.epochs // len(SQ_STAGES))
        for ratio in SQ_STAGES:
            set_sq_ratio(model, ratio)
            train_epochs(model, train_loader, test_loader, device, per_stage, args.lr,
                         f"[SQ {ratio:.1%}]")
        epochs_used = per_stage * len(SQ_STAGES)
    else:
        train_epochs(model, train_loader, test_loader, device, args.epochs, args.lr,
                     f"[{args.version}]")
        epochs_used = args.epochs

    # 3. Evaluate (in eval mode every filter is quantised) and save
    test_acc = evaluate(model, test_loader, device)
    print(f"{args.version} test accuracy: {test_acc:.2f}%")
    torch.save(model.state_dict(), os.path.join(run_dir, f"{args.version}.pt"))

    model.cpu()
    mode, bits = LOWBIT_VERSIONS[args.version]
    save_json({
        "model": args.model,
        "precision": args.version,
        "weight_bits": bits,
        "activation_bits": 8,
        "simulated": True,
        "seed": args.seed,
        "epochs": epochs_used,
        "from_scratch": args.from_scratch,
        "test_acc": test_acc,
        "params": count_params(model),
        "theoretical_size_mb": theoretical_size_mb(model),
    }, os.path.join(run_dir, f"{args.version}_metrics.json"))


if __name__ == "__main__":
    main()
