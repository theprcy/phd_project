"""
FGSM and PGD attacks (L-infinity). Images are in [0, 1], so eps = 8/255 means
each pixel may change by at most 8 brightness levels out of 255.

Both need gradients, so they work on FP32 models and QAT fake-quant models.
Real INT8 models have no gradients: they are attacked with images crafted on
another model (a transfer attack).
"""

import torch
import torch.nn.functional as F


def _input_gradient(model, x, y):
    """How the loss changes when each pixel changes."""
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    return grad


def fgsm(model, x, y, eps):
    """One step: move every pixel by eps in the direction that increases the loss."""
    model.eval()
    grad = _input_gradient(model, x, y)
    return (x + eps * grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, alpha, steps, random_start=True):
    """Many small FGSM steps of size alpha, kept within eps of the original image."""
    model.eval()
    x_adv = x.clone().detach()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        grad = _input_gradient(model, x_adv, y)
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)  # stay within eps
        x_adv = x_adv.clamp(0, 1).detach()                     # stay a valid image
    return x_adv


def run_attack(model, x, y, attack, eps, alpha=None, steps=10, device="cpu", batch_size=100):
    """Apply an attack to a whole tensor of images, in batches."""
    model = model.to(device)
    out = []
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        if attack == "fgsm":
            xa = fgsm(model, xb, yb, eps)
        elif attack == "pgd":
            xa = pgd(model, xb, yb, eps, alpha or eps / 4, steps)
        else:
            raise ValueError(f"Unknown attack '{attack}'")
        out.append(xa.cpu())
    return torch.cat(out)
