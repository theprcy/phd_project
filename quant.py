"""
Quantisation helpers (PyTorch eager-mode quantisation).

Backend:
  "x86"     - for testing on a normal laptop, desktop or Colab CPU
  "qnnpack" - for ARM processors such as the Raspberry Pi
Quantise with the backend of the device the model will run on.

PTQ steps: fuse layers -> insert observers -> run calibration images -> convert to INT8
QAT steps: fuse layers -> insert fake-quantisation -> fine-tune -> convert to INT8
"""

import torch
from torch.ao.quantization import (convert, disable_observer, get_default_qat_qconfig,
                                   get_default_qconfig, prepare, prepare_qat)


def set_backend(backend):
    torch.backends.quantized.engine = backend


def prepare_ptq(model, backend):
    """Fuse layers and add observers that record the range of values in each layer."""
    set_backend(backend)
    model.eval()
    model.fuse_model(qat=False)
    model.qconfig = get_default_qconfig(backend)
    prepare(model, inplace=True)
    return model


@torch.no_grad()
def calibrate(model, loader):
    """Run images through the model so the observers can record value ranges."""
    model.eval()
    for x, _ in loader:
        model(x)


def prepare_qat_model(model, backend):
    """Fuse layers and add fake-quantisation, so training 'feels' INT8 rounding."""
    set_backend(backend)
    model.train()
    model.fuse_model(qat=True)
    model.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(model, inplace=True)
    return model


def freeze_qat_ranges(model):
    """Stop updating the quantisation ranges (used for the last QAT epochs and for attacks)."""
    model.apply(disable_observer)
    model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)


def to_int8(model):
    """Convert a calibrated (PTQ) or trained (QAT) model to a real INT8 model (CPU only)."""
    model = model.cpu().eval()
    return convert(model, inplace=False)


def save_torchscript(model, path):
    """Save an INT8 model as TorchScript, so it can be loaded without the model code."""
    model.eval()
    traced = torch.jit.trace(model, torch.rand(1, 3, 32, 32))
    torch.jit.save(traced, path)
