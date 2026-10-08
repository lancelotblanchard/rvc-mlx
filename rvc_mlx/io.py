"""
On-disk format for converted models.

Every converted artefact is a single `.safetensors` file whose metadata header says what it is:

    format          "rvc-mlx"
    format_version  "1"
    kind            "voice" | "hubert" | "rmvpe"
    ...             kind-specific fields (JSON-encoded where structured)

Tensors are stored in MLX layout (channels-last convolution kernels, weight-norm already folded), so the Python,
Swift and C++ runtimes can all load them with a plain safetensors reader and no PyTorch. This module has no torch
dependency.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

FORMAT = "rvc-mlx"
FORMAT_VERSION = "1"
KINDS = ("voice", "hubert", "rmvpe")

ArrayLike = Union[mx.array, np.ndarray]


def save_converted(
    path: str,
    weights: Mapping[str, ArrayLike],
    kind: str,
    metadata: Optional[Mapping[str, str]] = None,
    dtype: Optional[mx.Dtype] = None,
) -> str:
    """
    Write `weights` + metadata to `path`. Floating-point tensors are cast to `dtype` when given (integer tensors are
    never cast). Returns `path`.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown kind {kind!r}; expected one of {KINDS}")
    arrays: Dict[str, mx.array] = {}
    for name, value in weights.items():
        arr = value if isinstance(value, mx.array) else mx.array(np.ascontiguousarray(value))
        if dtype is not None and mx.issubdtype(arr.dtype, mx.floating):
            arr = arr.astype(dtype)
        arrays[name] = arr
    meta = {"format": FORMAT, "format_version": FORMAT_VERSION, "kind": kind}
    for k, v in (metadata or {}).items():
        if not isinstance(v, str):
            raise TypeError(f"metadata values must be strings; {k!r} is {type(v).__name__}")
        meta[k] = v
    mx.save_safetensors(path, arrays, metadata=meta)
    return path


def read_metadata(path: str) -> Dict[str, str]:
    """Return the safetensors metadata header of `path` (empty for files written without one)."""
    _, meta = mx.load(path, return_metadata=True)
    return dict(meta or {})


def load_converted(
    path: str,
    expected_kind: Optional[str] = None,
    dtype: Optional[mx.Dtype] = mx.float32,
) -> Tuple[Dict[str, mx.array], Dict[str, str]]:
    """
    Load a converted file. Floating-point tensors are cast to `dtype` (pass `None` to keep the stored precision).

    Files produced before the metadata header existed (plain `save_weights` output) are accepted; they report an
    empty metadata dict and skip the `kind` check.
    """
    weights, meta = mx.load(path, return_metadata=True)
    meta = dict(meta or {})
    kind = meta.get("kind")
    if expected_kind is not None and kind is not None and kind != expected_kind:
        raise ValueError(f"{path!r} is a {kind!r} file, expected {expected_kind!r}")
    if dtype is not None:
        weights = {
            k: (v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in weights.items()
        }
    return weights, meta
