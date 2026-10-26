"""
Post-training quantisation (PTQ) of a trained FP32 model to INT8. No retraining.

Example:
    python scripts/ptq.py --model resnet20 --seed 0
    python scripts/ptq.py --model resnet20 --seed 0 --backend qnnpack   # for Raspberry Pi

Output: results/<model>_seed<seed>/ptq_int8_<backend>.pt (TorchScript) and metrics
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_calibration_loader, get_loaders
from src.models import build_model
from src.quant import calibrate, prepare_ptq, save_torchscript, to_int8
from src.utils import evaluate, file_size_mb, measure_latency_ms, save_json, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet20", "mobilenetv2"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["x86", "qnnpack"], default="x86")
    p.add_argument("--n-calib", type=int, default=1000, help="images used for calibration")
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./results")
    p.add_argument("--fake", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    run_dir = os.path.join(args.out, f"{args.model}_seed{args.seed}")

    # 1. Load the trained FP32 model (quantised models run on the CPU)
    model = build_model(args.model)
    model.load_state_dict(torch.load(os.path.join(run_dir, "fp32.pt"), map_location="cpu"))

    # 2. Fuse layers and add observers, 3. calibrate, 4. convert to INT8
    model = prepare_ptq(model, args.backend)
    calibrate(model, get_calibration_loader(args.data, args.n_calib, args.seed, args.fake))
    int8_model = to_int8(model)

    _, test_loader = get_loaders(args.data, seed=args.seed, fake=args.fake, num_workers=0)
    test_acc = evaluate(int8_model, test_loader, "cpu")
    print(f"PTQ INT8 test accuracy: {test_acc:.2f}%")

    path = os.path.join(run_dir, f"ptq_int8_{args.backend}.pt")
    save_torchscript(int8_model, path)
    save_json({
        "model": args.model,
        "precision": "ptq_int8",
        "backend": args.backend,
        "seed": args.seed,
        "n_calib": args.n_calib,
        "test_acc": test_acc,
        "size_mb": file_size_mb(path),
        "cpu_latency_ms": measure_latency_ms(int8_model),
    }, os.path.join(run_dir, f"ptq_int8_{args.backend}_metrics.json"))


if __name__ == "__main__":
    main()
