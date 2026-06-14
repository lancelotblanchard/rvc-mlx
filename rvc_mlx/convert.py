"""
Convert released RVC PyTorch checkpoints (RMVPE `rmvpe.pt` and per-voice synthesizer `.pth`) into MLX-native
safetensors files.

For RMVPE the conversion path is:

    .pt  --torch.load-->  state_dict  --load_state_dict-->  TorchE2E
                                                              |
                                                              v copy_e2e (transposes Conv weights, bridges GRU bias)
                                                            MLX E2E
                                                              |
                                                              v model.save_weights(out.safetensors)
                                                            out.safetensors

For the synthesizer (`SynthesizerTrnMs768NSFsid`) there are two additional twists:
  * The released checkpoint wraps `Conv1d`/`ConvTranspose1d` inside `nn.utils.weight_norm`, so the state_dict has
    `*.weight_g`/`*.weight_v` pairs instead of `*.weight`. We fuse those back into plain `weight` tensors before
    loading.
  * The training-time posterior encoder (`enc_q.*`) lives in the checkpoint but isn't used at inference; we strip
    those keys.

The synthesizer architecture config lives inside the released `.pth` (under the `config` key); we serialize it as a
sibling `*.config.json` next to the produced `*.safetensors` so `SynthesizerTrnMs768NSFsid.from_pretrained` can
rebuild the right architecture without the caller having to know the dimensions.

Torch is imported lazily — only consumers that actually convert pull it in.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple


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


def _fuse_weight_norm_state_dict(state_dict: Dict) -> Dict:
    """
    Fuse `weight_norm`-style parameter pairs back into plain `weight` tensors.

    `nn.utils.weight_norm(module, name="weight")` removes the original `weight` parameter and adds two new ones:
        * `weight_g`: per-output-channel norm (shape `(out_channels, 1, ..., 1)`)
        * `weight_v`: unnormalized weight (same shape as the original `weight`)
    The forward pass reconstructs `weight = weight_g * weight_v / ||weight_v||` where the norm is taken over all axes
    except dim 0. To load a published RVC checkpoint into our plain-`Conv1d` modules we apply that same formula here.

    Any keys not matching the `weight_g`/`weight_v` pattern pass through unchanged.
    """
    # Lazy torch import: only callers that actually convert pull torch in.
    import torch

    out: Dict = {}
    pending: Dict[str, Dict[str, "torch.Tensor"]] = {}
    for k, v in state_dict.items():
        if k.endswith(".weight_g"):
            pending.setdefault(k[: -len(".weight_g")], {})["g"] = v
        elif k.endswith(".weight_v"):
            pending.setdefault(k[: -len(".weight_v")], {})["v"] = v
        else:
            out[k] = v

    for base, parts in pending.items():
        if "g" not in parts or "v" not in parts:
            raise ValueError(
                f"Incomplete weight_norm pair for {base!r}: expected both .weight_g and .weight_v."
            )
        g = parts["g"]
        v = parts["v"]
        # Norm along all axes except dim 0 (matches `weight_norm`'s default `dim=0`).
        reduce_axes = list(range(1, v.ndim))
        v_norm = torch.linalg.vector_norm(v, dim=reduce_axes, keepdim=True)
        out[base + ".weight"] = g * v / v_norm
    return out


# Position-to-name mapping for the synthesizer config list stored in released `.pth` checkpoints. The list ordering
# matches `SynthesizerTrnMs768NSFsid.__init__`'s argument order.
_SYNTH_CONFIG_KEYS = (
    "spec_channels",
    "segment_size",
    "inter_channels",
    "hidden_channels",
    "filter_channels",
    "n_heads",
    "n_layers",
    "kernel_size",
    "p_dropout",
    "resblock",
    "resblock_kernel_sizes",
    "resblock_dilation_sizes",
    "upsample_rates",
    "upsample_initial_channel",
    "upsample_kernel_sizes",
    "spk_embed_dim",
    "gin_channels",
    "sr",
)


def _normalize_synth_config(raw) -> Dict:
    """
    The synthesizer config in a released `.pth` is a list (positional) or a dict. Return a kwargs dict suitable for
    `SynthesizerTrnMs768NSFsid.__init__`.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        if len(raw) != len(_SYNTH_CONFIG_KEYS):
            raise ValueError(
                f"Expected synthesizer config list of length {len(_SYNTH_CONFIG_KEYS)}, got {len(raw)}."
            )
        return dict(zip(_SYNTH_CONFIG_KEYS, raw))
    raise TypeError(
        f"Unsupported synthesizer config type: {type(raw).__name__}. Expected list, tuple, or dict."
    )


def convert_synthesizer_checkpoint(
    pt_path: str,
    out_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Convert a single RVC `SynthesizerTrnMs768NSFsid` PyTorch checkpoint to MLX safetensors + config JSON.

    :param pt_path: path to the released `.pth` file. The checkpoint must be a dict containing at minimum
        `weight` (the state_dict) and `config` (the architecture hyperparameters; list or dict form both accepted).
    :param out_path: where to write the safetensors file. Defaults to `<pt_path stem>.safetensors` next to the input.
        The config is always written to a sibling `<stem>.config.json` (this isn't configurable to keep the load path
        single-argument).
    :returns: a tuple `(safetensors_path, config_json_path)`.
    """
    import torch
    from rvc_mlx._torch_ref import TorchSynthesizerTrnMs768NSFsid
    from rvc_mlx._convert_bridge import copy_synthesizer_trn_ms768_nsfsid, set_eval
    from rvc_mlx.synthesizer import SynthesizerTrnMs768NSFsid

    if out_path is None:
        stem, _ = os.path.splitext(pt_path)
        out_path = stem + ".safetensors"
    config_path = out_path[: -len(".safetensors")] + ".config.json"

    cpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(
            f"Unsupported RVC checkpoint at {pt_path!r}: expected top-level 'config' and 'weight' keys, got "
            f"{sorted(cpt.keys()) if isinstance(cpt, dict) else type(cpt).__name__}."
        )
    config = _normalize_synth_config(cpt["config"])

    raw_state_dict = cpt["weight"]
    # Fuse weight_norm parameter pairs and drop the training-only posterior encoder.
    state_dict = _fuse_weight_norm_state_dict(raw_state_dict)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("enc_q.")}

    torch_model = TorchSynthesizerTrnMs768NSFsid(**config)
    # `strict=False` so any other auxiliary keys (e.g. `emb_g.weight` from a possibly-different version) are tolerated;
    # we still verify the missing/unexpected lists below.
    incompatible = torch_model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        raise ValueError(
            f"Checkpoint at {pt_path!r} is missing required keys: {incompatible.missing_keys[:5]}..."
        )

    mlx_model = SynthesizerTrnMs768NSFsid(**config)
    copy_synthesizer_trn_ms768_nsfsid(torch_model, mlx_model)
    set_eval(torch_model, mlx_model)

    mlx_model.save_weights(out_path)
    with open(config_path, "w") as f:
        # Cast tuples to lists for JSON serialization.
        json.dump(_json_safe_config(config), f, indent=2)

    return out_path, config_path


def _json_safe_config(config: Dict) -> Dict:
    """Convert tuples (used internally for hashability) to lists for JSON output."""

    def _convert(v):
        if isinstance(v, tuple):
            return [_convert(x) for x in v]
        if isinstance(v, list):
            return [_convert(x) for x in v]
        return v

    return {k: _convert(v) for k, v in config.items()}


def ensure_synthesizer_safetensors(path: str) -> Tuple[str, Dict]:
    """
    Return `(safetensors_path, config)` for a synthesizer checkpoint. Converts on first access if given a `.pth`.

    The config is loaded from the sibling `*.config.json` (produced by `convert_synthesizer_checkpoint`); for a
    `.safetensors` input that JSON must already exist next to it.
    """
    if path.endswith(".safetensors"):
        config_path = path[: -len(".safetensors")] + ".config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Synthesizer safetensors {path!r} is missing its sibling config at {config_path!r}. "
                "Re-run `convert_synthesizer_checkpoint` to regenerate."
            )
        with open(config_path) as f:
            config = json.load(f)
        return path, config

    if path.endswith(".pth") or path.endswith(".pt"):
        stem = path.rsplit(".", 1)[0]
        cached = stem + ".safetensors"
        config_path = stem + ".config.json"
        if os.path.exists(cached) and os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            return cached, config
        cached, config_path = convert_synthesizer_checkpoint(path, cached)
        with open(config_path) as f:
            config = json.load(f)
        return cached, config

    raise ValueError(
        f"Unsupported synthesizer checkpoint extension for {path!r}: expected .safetensors, .pth, or .pt."
    )


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
