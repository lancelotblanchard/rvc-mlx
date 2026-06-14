"""
Convert released RVC RMVPE PyTorch checkpoints (`rmvpe.pt`) into MLX-native safetensors files.

The released RVC RMVPE checkpoint is a `state_dict` saved against the `E2E` reference module in
`rvc_mlx._torch_ref`. The conversion path is:

    .pt  --torch.load-->  state_dict  --load_state_dict-->  TorchE2E
                                                              |
                                                              v copy_e2e (transposes Conv weights, bridges GRU bias)
                                                            MLX E2E
                                                              |
                                                              v model.save_weights(out.safetensors)
                                                            out.safetensors

Torch is imported lazily — only consumers that actually convert pull it in. The inference path
(`rvc_mlx.rmvpe.RMVPE.from_pretrained`) calls into here only when given a `.pt` input, so a runtime that ships only
`.safetensors` files never imports torch.
"""

from __future__ import annotations

import os
from typing import Optional


def convert_rmvpe_checkpoint(
    pt_path: str,
    out_path: Optional[str] = None,
    is_half: bool = False,
) -> str:
    """
    Convert a single RVC RMVPE PyTorch checkpoint to MLX safetensors.

    :param pt_path: path to the released `.pt` file (a `state_dict` matching `rvc_mlx._torch_ref.TorchE2E`).
    :param out_path: where to write the safetensors file. Defaults to `<pt_path stem>.safetensors` next to the input.
    :param is_half: ignored at conversion time (weights are stored at their loaded precision); kept for symmetry
        with the inference-side `from_pretrained` so callers can pass the same flag through.
    :returns: the resolved `out_path`.
    """
    # Lazy imports so importing this module without performing a conversion doesn't drag torch in.
    import torch  # noqa: F401 — sanity check the dep exists with a clear error
    from rvc_mlx._torch_ref import TorchE2E, DEFAULT_E2E_CONFIG
    from rvc_mlx._convert_bridge import copy_e2e, set_eval
    from rvc_mlx.rmvpe import E2E

    if out_path is None:
        stem, _ = os.path.splitext(pt_path)
        out_path = stem + ".safetensors"

    # Released RVC checkpoints store the model `state_dict` directly (no optimizer / training-state wrapping).
    # If a future release nests it, swap this for `state_dict["model"]` or similar.
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

    torch_model = TorchE2E(**DEFAULT_E2E_CONFIG)
    torch_model.load_state_dict(state_dict)

    mlx_model = E2E(**DEFAULT_E2E_CONFIG)
    copy_e2e(torch_model, mlx_model)
    # Eval mode pins BatchNorm to running stats. We already copied the running stats; `set_eval` makes sure the
    # serialized MLX state reflects that mode if the file is loaded by code that respects the train/eval flag.
    set_eval(torch_model, mlx_model)

    mlx_model.save_weights(out_path)
    return out_path


def ensure_safetensors(path: str, is_half: bool = False) -> str:
    """
    Return the path to a safetensors checkpoint, converting from `.pt` on demand.

    The first time a user points at a `.pt`, this writes a sibling `.safetensors` and returns its path. Subsequent
    calls reuse the cached file — no torch needed once the conversion has happened.
    """
    if path.endswith(".safetensors"):
        return path
    if path.endswith(".pt"):
        cached = path[: -len(".pt")] + ".safetensors"
        if os.path.exists(cached):
            return cached
        return convert_rmvpe_checkpoint(path, cached, is_half=is_half)
    raise ValueError(f"Unsupported checkpoint extension for {path!r}: expected .safetensors or .pt")
