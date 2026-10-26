"""
Quantisation-aware training (QAT): start from the trained FP32 model, fine-tune it
while simulating INT8 rounding ("fake quantisation"), then convert to real INT8.

Example:
    python scripts/qat.py --model resnet20 --seed 0 --epochs 15

Output in results/<model>_seed<seed>/:
    qat_fakequant_<backend>.pt  - fake-quant weights (has gradients: used for white-box attacks)
    qat_int8_<backend>.pt       - real INT8 model as TorchScript (for deployment)
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import train_one_epoch
from src.data import get_loaders
from src.models import build_model
from src.quant import freeze_qat_ranges, prepare_qat_model, save_torchscript, to_int8
from src.utils import (evaluate, file_size_mb, get_device, measure_latency_ms, save_json,
                       set_seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet20", "mobilenetv2"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["x86", "qnnpack"], default="x86")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.01, help="small: we fine-tune, not retrain")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./results")
    p.add_argument("--fake", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    run_dir = os.path.join(args.out, f"{args.model}_seed{args.seed}")

    # 1. Start from the trained FP32 model
    model = build_model(args.model)
    model.load_state_dict(torch.load(os.path.join(run_dir, "fp32.pt"), map_location="cpu"))

    # 2. Fuse layers and insert fake quantisation
    model = prepare_qat_model(model, args.backend).to(device)

    # 3. Fine-tune. For the last 30% of epochs, freeze the quantisation ranges and
    #    batchnorm statistics so the model settles on fixed INT8 settings.
    train_loader, test_loader = get_loaders(args.data, args.batch_size, seed=args.seed,
                                            fake=args.fake)
    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    # (never at epoch 1: the observers must first see real data to set the ranges)
    freeze_epoch = int(0.7 * args.epochs) + 1
    for epoch in range(1, args.epochs + 1):
        if epoch == freeze_epoch and epoch > 1:
            freeze_qat_ranges(model)
            print("Froze quantisation ranges and batchnorm statistics")
        loss, train_acc = train_one_epoch(model, train_loader, optimiser, device)
        scheduler.step()
        print(f"epoch {epoch:3d}  loss {loss:.3f}  train {train_acc:.2f}%")
    freeze_qat_ranges(model)

    fq_acc = evaluate(model, test_loader, device)
    print(f"QAT fake-quant test accuracy: {fq_acc:.2f}%")
    torch.save(model.state_dict(), os.path.join(run_dir, f"qat_fakequant_{args.backend}.pt"))

    # 4. Convert to real INT8
    int8_model = to_int8(model)
    _, test_loader_cpu = get_loaders(args.data, seed=args.seed, fake=args.fake, num_workers=0)
    test_acc = evaluate(int8_model, test_loader_cpu, "cpu")
    print(f"QAT INT8 test accuracy: {test_acc:.2f}%")

    path = os.path.join(run_dir, f"qat_int8_{args.backend}.pt")
    save_torchscript(int8_model, path)
    save_json({
        "model": args.model,
        "precision": "qat_int8",
        "backend": args.backend,
        "seed": args.seed,
        "qat_epochs": args.epochs,
        "fakequant_test_acc": fq_acc,
        "test_acc": test_acc,
        "size_mb": file_size_mb(path),
        "cpu_latency_ms": measure_latency_ms(int8_model),
    }, os.path.join(run_dir, f"qat_int8_{args.backend}_metrics.json"))


if __name__ == "__main__":
    main()
