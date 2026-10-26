"""
Simulated low-bit quantisation ("fake quantisation") for:
  - QAT 4-bit  : weights rounded to 16 levels per filter        (mode "uniform", bits=4)
  - TWN        : weights become -a, 0 or +a per filter          (mode "ternary")
  - SQ-TWN     : TWN, but only some filters quantised at a time (mode "sq"), Dong et al. (2017)

Activations (each layer's input) are fake-quantised to 8 bits in all modes.
The first and last layers keep 8-bit weights, which is common practice.

These models only SIMULATE low-bit weights: they run in float, so accuracy and
robustness are real, but size and speed savings are theoretical.

Straight-through estimator (STE): rounding has zero gradient, so we use
    w + (q - w).detach()
which gives the quantised value q in the forward pass, but passes the gradient
straight to w in the backward pass, as if no rounding had happened.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import build_model


# ---------------------------------------------------------------- weight quantisers

def _per_filter(w):
    """Reshape weights so each row is one filter (one output channel)."""
    return w.reshape(w.shape[0], -1)


def _broadcast(v, w):
    """Reshape a per-filter value so it can be multiplied with w."""
    return v.view(-1, *[1] * (w.dim() - 1))


def quantise_uniform(w, bits):
    """Symmetric, per-filter: each filter gets its own scale.
    4 bits -> integer levels -7..+7, scale = largest |weight| in the filter / 7."""
    qmax = 2 ** (bits - 1) - 1
    scale = _per_filter(w).abs().max(dim=1).values.clamp(min=1e-8) / qmax
    scale = _broadcast(scale, w)
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale


def quantise_ternary(w):
    """TWN (Li et al., 2016), per filter:
    threshold = 0.7 x average |weight|; weights above it -> +1, below -it -> -1, rest -> 0;
    scale a = average |weight| of the weights that were not set to 0."""
    flat = _per_filter(w)
    delta = 0.7 * flat.abs().mean(dim=1, keepdim=True)
    keep = (flat.abs() > delta).float()
    alpha = (flat.abs() * keep).sum(dim=1, keepdim=True) / keep.sum(dim=1, keepdim=True).clamp(min=1)
    return (alpha * torch.sign(flat) * keep).view_as(w)


def sq_select(w, q, ratio):
    """Stochastic quantisation (Dong et al., 2017): choose which filters to quantise.

    1. error per filter = sum|w - q| / sum|w|   (normalised L1 distance, their eq. 6)
    2. f = 1 / error                              (small error -> large f)
    3. probability = f / sum(f)                   (their "linear" function, the best one)
    4. pick ratio x (number of filters) at random with these probabilities,
       without picking the same filter twice      (their "roulette" algorithm)
    Returns a mask: 1 = use quantised weights, 0 = keep full precision."""
    m = w.shape[0]
    n_quantised = int(round(ratio * m))
    mask = torch.zeros(m, device=w.device)
    if n_quantised == 0:
        return mask
    err = _per_filter(w - q).abs().sum(dim=1) / _per_filter(w).abs().sum(dim=1).clamp(min=1e-8)
    f = 1.0 / (err + 1e-7)
    p = f / f.sum()
    chosen = torch.multinomial(p, n_quantised, replacement=False)
    mask[chosen] = 1.0
    return mask


# ---------------------------------------------------------------- activation quantiser

class ActQuant(nn.Module):
    """8-bit fake quantisation of a layer's input.
    During training the value range is tracked with a moving average; in eval it is fixed."""

    def __init__(self, bits=8, momentum=0.1):
        super().__init__()
        self.bits, self.momentum = bits, momentum
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(0.0))
        self.register_buffer("ready", torch.tensor(False))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                lo, hi = x.min(), x.max()
                if not self.ready:
                    self.lo.copy_(lo)
                    self.hi.copy_(hi)
                    self.ready.fill_(True)
                else:
                    self.lo.mul_(1 - self.momentum).add_(self.momentum * lo)
                    self.hi.mul_(1 - self.momentum).add_(self.momentum * hi)
        if not self.ready:
            return x
        scale = (self.hi - self.lo).clamp(min=1e-8) / (2 ** self.bits - 1)
        q = torch.round((x.clamp(self.lo, self.hi) - self.lo) / scale) * scale + self.lo
        return x + (q - x).detach()  # STE


# ---------------------------------------------------------------- low-bit layers

class _LowBit:
    """Shared code for low-bit Conv2d and Linear layers."""

    def setup(self, mode, bits, act_bits=8):
        self.mode, self.bits = mode, bits
        self.sq_ratio = 1.0  # SQ only: fraction of filters quantised
        self.act_quant = ActQuant(act_bits)

    def quantised_weight(self):
        w = self.weight
        q = quantise_uniform(w, self.bits) if self.mode == "uniform" else quantise_ternary(w)
        wq = w + (q - w).detach()  # STE
        if self.mode == "sq" and self.training and self.sq_ratio < 1.0:
            mask = _broadcast(sq_select(w.detach(), q.detach(), self.sq_ratio), w)
            return mask * wq + (1 - mask) * w  # hybrid: some filters quantised, rest full precision
        return wq  # in eval (and once ratio = 1), every filter is quantised

    def stored_bits(self):
        return 2 if self.mode in ("ternary", "sq") else self.bits


class LowBitConv2d(_LowBit, nn.Conv2d):
    def forward(self, x):
        return self._conv_forward(self.act_quant(x), self.quantised_weight(), self.bias)


class LowBitLinear(_LowBit, nn.Linear):
    def forward(self, x):
        return F.linear(self.act_quant(x), self.quantised_weight(), self.bias)


def _to_lowbit(m, mode, bits, act_bits):
    if isinstance(m, nn.Conv2d):
        new = LowBitConv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding,
                           m.dilation, m.groups, m.bias is not None)
    else:
        new = LowBitLinear(m.in_features, m.out_features, m.bias is not None)
    new.setup(mode, bits, act_bits)
    new.weight.data.copy_(m.weight.data)
    if m.bias is not None:
        new.bias.data.copy_(m.bias.data)
    return new.to(m.weight.device)


def convert_to_lowbit(model, mode, bits=4, act_bits=8, first_last_8bit=True):
    """Replace every Conv2d and Linear layer with its low-bit version (keeps current weights)."""
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    first, last = layers[0][0], layers[-1][0]
    for name, m in layers:
        if first_last_8bit and name in (first, last):
            new = _to_lowbit(m, "uniform", 8, act_bits)
        else:
            new = _to_lowbit(m, mode, bits, act_bits)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, new)
    return model


# The three low-bit versions used in Stage 1: name -> (mode, bits)
LOWBIT_VERSIONS = {"qat4": ("uniform", 4), "twn": ("ternary", 2), "sqtwn": ("sq", 2)}


def build_lowbit_model(model_name, version):
    mode, bits = LOWBIT_VERSIONS[version]
    return convert_to_lowbit(build_model(model_name), mode, bits)


def set_sq_ratio(model, ratio):
    for m in model.modules():
        if isinstance(m, _LowBit):
            m.sq_ratio = ratio


def theoretical_size_mb(model):
    """Size if low-bit weights were stored with their true number of bits
    (plus one 32-bit scale per filter); everything else stays 32-bit."""
    total_bits = 0
    lowbit_params = set()
    for m in model.modules():
        if isinstance(m, _LowBit):
            total_bits += m.weight.numel() * m.stored_bits() + m.weight.shape[0] * 32
            lowbit_params.add(id(m.weight))
    for p in model.parameters():
        if id(p) not in lowbit_params:
            total_bits += p.numel() * 32
    return total_bits / 8 / 1e6
