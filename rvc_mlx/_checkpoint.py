"""
Helpers for reading PyTorch checkpoints during conversion. Imports torch; never import this from inference code.
"""

from __future__ import annotations

import pickle
import re
from typing import Dict, Mapping

import mlx.nn as nn
import numpy as np


class _Opaque:
    """Stand-in for any pickled class we refuse to import (fairseq/omegaconf configs and the like)."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.state = state

    def __call__(self, *args, **kwargs):  # pickled callables (e.g. enum lookups) resolve to another opaque value
        return _Opaque(*args, **kwargs)


_SAFE_BUILTINS = {"dict", "list", "tuple", "set", "frozenset", "int", "float", "bool", "str", "bytes", "slice",
                  "complex", "bytearray", "object"}
_SAFE_TORCH_UTILS = {"_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_parameter", "_rebuild_parameter_with_state"}


class _LenientUnpickler(pickle.Unpickler):
    """
    Unpickler that only materialises tensors and plain containers. Unknown classes (training configs from fairseq,
    omegaconf, argparse, ...) are replaced by inert `_Opaque` objects instead of being imported, so converting a
    checkpoint never needs fairseq installed and never executes arbitrary pickled callables.
    """

    def find_class(self, module, name):
        if (module, name) in (("collections", "OrderedDict"), ("argparse", "Namespace")):
            return super().find_class(module, name)
        if module == "builtins" and name in _SAFE_BUILTINS:
            return super().find_class(module, name)
        if module == "torch._utils" and name in _SAFE_TORCH_UTILS:
            return super().find_class(module, name)
        if module == "torch" and (name.endswith("Storage") or name in ("Size", "device") or name.startswith(("float", "int", "bfloat", "half", "bool", "uint"))):
            return super().find_class(module, name)
        if module == "numpy.core.multiarray" and name in ("_reconstruct", "scalar"):
            return super().find_class(module, name)
        if module in ("numpy", "numpy.dtype") and name in ("ndarray", "dtype"):
            return super().find_class(module, name)
        if module.startswith("numpy.dtypes"):
            return super().find_class(module, name)
        return _Opaque


class _LenientPickleModule:
    Unpickler = _LenientUnpickler
    load = staticmethod(pickle.load)
    __name__ = "rvc_mlx_lenient_pickle"


def load_torch_checkpoint(path: str):
    """
    Load a PyTorch checkpoint on CPU. Tries the strict `weights_only` loader first (fine for RVC voice `.pth` files);
    falls back to a restricted unpickler for fairseq checkpoints such as `hubert_base.pt`.
    """
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_LenientPickleModule)


def to_numpy_state_dict(state: Mapping) -> Dict[str, np.ndarray]:
    """Torch tensors / numpy arrays -> float32 (or integer) numpy arrays; other entries are dropped."""
    try:
        import torch
    except ImportError:  # safetensors inputs can be converted without torch
        torch = None

    out = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            out[k] = v.astype(np.float32) if np.issubdtype(v.dtype, np.floating) else v
        elif torch is not None and isinstance(v, torch.Tensor):
            v = v.detach().cpu()
            if v.is_floating_point():
                v = v.float()
            out[k] = v.numpy()
    return out


def _fold(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    # `torch.nn.utils.weight_norm(dim=d)` stores g with size 1 on every axis except d; the norm of v is taken over
    # exactly those size-1 axes. Inferring the axes from g's shape handles dim=0 (default) and dim=2 (HuBERT).
    axes = tuple(i for i, s in enumerate(g.shape) if s == 1)
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=axes, keepdims=True))
    return (g * v / norm).astype(np.float32)


_PARAM_G = re.compile(r"^(.*)\.parametrizations\.weight\.original0$")


def fold_weight_norm(state: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Replace weight-normalised parameter pairs with the effective weight. Handles both the legacy hook naming
    (`*.weight_g` / `*.weight_v`) and the parametrization naming (`*.parametrizations.weight.original0/1`).
    """
    out: Dict[str, np.ndarray] = {}
    for k, v in state.items():
        if k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            out[base + ".weight"] = _fold(v, state[base + ".weight_v"])
        elif k.endswith(".weight_v"):
            continue
        elif (m := _PARAM_G.match(k)) is not None:
            base = m.group(1)
            out[base + ".weight"] = _fold(v, state[base + ".parametrizations.weight.original1"])
        elif k.endswith(".parametrizations.weight.original1"):
            continue
        else:
            out[k] = v
    return out


def to_mlx_layout(model: nn.Module, state: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Transpose convolution kernels from PyTorch layout to MLX channels-last layout, deciding per parameter from the
    type of the MLX module that owns it. Keys must already use the MLX module's naming.
    """
    kinds = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.ConvTranspose1d):
            kinds[name] = (1, 2, 0)
        elif isinstance(module, nn.Conv1d):
            kinds[name] = (0, 2, 1)
        elif isinstance(module, nn.ConvTranspose2d):
            kinds[name] = (1, 2, 3, 0)
        elif isinstance(module, nn.Conv2d):
            kinds[name] = (0, 2, 3, 1)
    out = {}
    for k, v in state.items():
        owner, _, leaf = k.rpartition(".")
        perm = kinds.get(owner)
        if perm is not None and leaf == "weight":
            v = np.transpose(v, perm)
        out[k] = np.ascontiguousarray(v)
    return out
