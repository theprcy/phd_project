"""
Attack every saved version of a model and record accuracy, attack success rate (ASR),
confidence reliability and a gradient masking check (RQ1).

Example:
    python scripts/attack.py --model resnet20 --seed 0 --attack pgd  --eps 2/255,4/255,8/255
    python scripts/attack.py --model resnet20 --seed 0 --attack fgsm --eps 2/255,4/255,8/255

Versions attacked (whichever exist in results/<model>_seed<seed>/):
    fp32, qat_fakequant, ptq_int8, qat_int8, qat4, twn, sqtwn

White-box = attack crafted on the model itself (needs gradients).
Transfer  = attack crafted on the FP32 model, then applied (works for every version).
Masking gap = transfer ASR - white-box ASR. A clearly positive gap means the model's own
gradients are a poor guide for attacks, which suggests gradient masking, not real robustness.

Output: results/<model>_seed<seed>/attack_<attack>_<backend>.json
"""

import argparse
import os
import sys
from fractions import Fraction

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attacks import run_attack
from src.confidence_metrics import attack_report, confidence_report, print_table
from src.data import get_test_subset
from src.lowbit import LOWBIT_VERSIONS, build_lowbit_model
from src.models import build_model
from src.quant import freeze_qat_ranges, prepare_qat_model, set_backend
from src.utils import get_device, predict_probs, save_json, set_seed


def load_models(model_name, run_dir, backend, device):
    """Returns {name: (model, device, has_gradients)} for every version found."""
    models = {}
    fp32 = build_model(model_name)
    fp32.load_state_dict(torch.load(os.path.join(run_dir, "fp32.pt"), map_location="cpu"))
    models["fp32"] = (fp32.to(device).eval(), device, True)

    path = os.path.join(run_dir, f"qat_fakequant_{backend}.pt")
    if os.path.exists(path):
        fq = prepare_qat_model(build_model(model_name), backend)
        fq.load_state_dict(torch.load(path, map_location="cpu"))
        freeze_qat_ranges(fq)
        models["qat_fakequant"] = (fq.to(device).eval(), device, True)

    set_backend(backend)
    for name in ("ptq_int8", "qat_int8"):
        path = os.path.join(run_dir, f"{name}_{backend}.pt")
        if os.path.exists(path):
            models[name] = (torch.jit.load(path, map_location="cpu").eval(), "cpu", False)

    for name in LOWBIT_VERSIONS:
        path = os.path.join(run_dir, f"{name}.pt")
        if os.path.exists(path):
            m = build_lowbit_model(model_name, name)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            models[name] = (m.to(device).eval(), device, True)

    print("Versions found:", ", ".join(models))
    return models


def to_float(d):
    """NumPy numbers -> plain floats, so results can be saved as JSON."""
    if isinstance(d, dict):
        return {k: to_float(v) for k, v in d.items()}
    return float(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet20", "mobilenetv2"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["x86", "qnnpack"], default="x86")
    p.add_argument("--attack", choices=["fgsm", "pgd"], default="pgd")
    p.add_argument("--eps", default="2/255,4/255,8/255", help="comma-separated attack sizes")
    p.add_argument("--steps", type=int, default=10, help="PGD steps (step size = eps/4)")
    p.add_argument("--n", type=int, default=1000, help="number of test images to attack")
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./results")
    p.add_argument("--fake", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    run_dir = os.path.join(args.out, f"{args.model}_seed{args.seed}")

    x, y = get_test_subset(args.data, args.n, args.seed, args.fake)
    labels = y.numpy()
    models = load_models(args.model, run_dir, args.backend, device)

    # Clean predictions, computed once per version
    clean = {name: predict_probs(m, x, dev) for name, (m, dev, _) in models.items()}

    all_results = {}
    for eps_str in args.eps.split(","):
        eps = float(Fraction(eps_str))
        print(f"\n===== {args.attack.upper()}  eps = {eps_str} =====")
        fp32_model, fp32_dev, _ = models["fp32"]
        transfer_images = run_attack(fp32_model, x, y, args.attack, eps, eps / 4, args.steps,
                                     fp32_dev)

        results, table = {}, {}
        for name, (model, dev, has_grad) in models.items():
            r = {"clean": confidence_report(clean[name], labels)}
            table[f"{name} clean"] = r["clean"]

            # transfer attack (for fp32 this is the same as white-box)
            adv = predict_probs(model, transfer_images, dev)
            r["transfer"] = confidence_report(adv, labels)
            r["transfer_asr"] = attack_report(clean[name], adv, labels)["asr"]
            table[f"{name} transfer"] = r["transfer"]

            # white-box attack on the version's own gradients
            if has_grad and name != "fp32":
                wb_images = run_attack(model, x, y, args.attack, eps, eps / 4, args.steps, dev)
                adv = predict_probs(model, wb_images, dev)
                r["whitebox"] = confidence_report(adv, labels)
                r["whitebox_asr"] = attack_report(clean[name], adv, labels)["asr"]
                r["masking_gap"] = r["transfer_asr"] - r["whitebox_asr"]
                table[f"{name} white-box"] = r["whitebox"]
            elif name == "fp32":
                r["whitebox_asr"] = r["transfer_asr"]

            msg = f"{name:<14} transfer ASR {100 * r['transfer_asr']:5.1f}%"
            if "masking_gap" in r:
                msg += (f"   white-box ASR {100 * r['whitebox_asr']:5.1f}%"
                        f"   masking gap {100 * r['masking_gap']:+5.1f}")
            print(msg)
            results[name] = r

        print()
        print_table(table)
        all_results[eps_str] = results

    save_json({"model": args.model, "seed": args.seed, "attack": args.attack,
               "steps": args.steps, "n": args.n, "backend": args.backend,
               "results": to_float(all_results)},
              os.path.join(run_dir, f"attack_{args.attack}_{args.backend}.json"))


if __name__ == "__main__":
    main()
