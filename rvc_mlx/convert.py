"""
Convert released RVC PyTorch checkpoints into MLX-native safetensors files (see `rvc_mlx.io` for the format).

Three artefacts make up an RVC setup:

    rmvpe.pt          -> rmvpe.safetensors    pitch estimator (shared by every voice)
    hubert_base.pt    -> hubert.safetensors   content encoder (shared by every voice)
    <voice>.pth (+ .index) -> <voice>.safetensors   one per voice

Torch is imported lazily: only callers that actually convert pull it in. The inference path loads `.safetensors` only.

The RMVPE path goes through the paired PyTorch reference module (`rvc_mlx._torch_ref`) and the copy helpers in
`rvc_mlx._convert_bridge`; HuBERT and voices are converted directly from their state dicts by renaming keys, folding
weight norm and transposing convolution kernels to channels-last.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

import mlx.core as mx
import numpy as np

_DTYPES = {"float16": mx.float16, "float32": mx.float32, "bfloat16": mx.bfloat16}


def _resolve_dtype(dtype) -> Optional[mx.Dtype]:
    if dtype is None or isinstance(dtype, mx.Dtype):
        return dtype
    try:
        return _DTYPES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(_DTYPES)}") from None


def _default_out(path: str, out_path: Optional[str]) -> str:
    if out_path is not None:
        return out_path
    stem, _ = os.path.splitext(path)
    return stem + ".safetensors"


# ---------------------------------------------------------------------------------------------------------------------
# RMVPE
# ---------------------------------------------------------------------------------------------------------------------


def convert_rmvpe_checkpoint(
    pt_path: str,
    out_path: Optional[str] = None,
    is_half: bool = False,
    dtype=None,
) -> str:
    """
    Convert a single RVC RMVPE PyTorch checkpoint to MLX safetensors.

    :param pt_path: path to the released `.pt` file (a `state_dict` matching `rvc_mlx._torch_ref.TorchE2E`).
    :param out_path: where to write the safetensors file. Defaults to `<pt_path stem>.safetensors` next to the input.
    :param is_half: ignored at conversion time; kept for symmetry with the inference-side `from_pretrained`.
    :param dtype: storage precision (`"float16"`, `"float32"`, ...). Defaults to the checkpoint's own (float32).
    :returns: the resolved `out_path`.
    """
    # Lazy imports so importing this module without performing a conversion doesn't drag torch in.
    import torch
    from mlx.utils import tree_flatten

    from rvc_mlx import rmvpe as rmvpe_mod
    from rvc_mlx._convert_bridge import copy_e2e, set_eval
    from rvc_mlx._torch_ref import TorchE2E
    from rvc_mlx.io import save_converted
    from rvc_mlx import _torch_ref

    out_path = _default_out(pt_path, out_path)
    config = _torch_ref.DEFAULT_E2E_CONFIG

    # Released RVC checkpoints store the model `state_dict` directly (no optimizer / training-state wrapping).
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

    torch_model = TorchE2E(**config)
    torch_model.load_state_dict(state_dict)

    mlx_model = rmvpe_mod.E2E(**config)
    copy_e2e(torch_model, mlx_model)
    set_eval(torch_model, mlx_model)

    weights = dict(tree_flatten(mlx_model.parameters()))
    # The mel filterbank is a fixed function of the RMVPE front-end settings. Storing it next to the weights lets the
    # Swift and C++ runtimes skip re-implementing `librosa.filters.mel(htk=True)`.
    weights[rmvpe_mod.MEL_BASIS_KEY] = rmvpe_mod.rmvpe_mel_basis()
    save_converted(
        out_path,
        weights,
        kind="rmvpe",
        metadata={"config": _json(config)},
        dtype=_resolve_dtype(dtype),
    )
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


# ---------------------------------------------------------------------------------------------------------------------
# HuBERT / ContentVec
# ---------------------------------------------------------------------------------------------------------------------

# Hugging Face `transformers` HubertModel naming -> our (fairseq-derived) naming. Applied after weight-norm folding.
_HF_HUBERT_RULES = [
    (r"^hubert\.", ""),
    (r"^feature_extractor\.conv_layers\.0\.layer_norm\.", "feature_extractor.conv_layers.0.norm."),
    (r"^feature_projection\.layer_norm\.", "layer_norm."),
    (r"^feature_projection\.projection\.", "post_extract_proj."),
    (r"^encoder\.pos_conv_embed\.conv\.", "encoder.pos_conv."),
    (r"^encoder\.layers\.(\d+)\.attention\.", r"encoder.layers.\1.self_attn."),
    (r"^encoder\.layers\.(\d+)\.layer_norm\.", r"encoder.layers.\1.self_attn_layer_norm."),
    (r"^encoder\.layers\.(\d+)\.feed_forward\.intermediate_dense\.", r"encoder.layers.\1.fc1."),
    (r"^encoder\.layers\.(\d+)\.feed_forward\.output_dense\.", r"encoder.layers.\1.fc2."),
]
# fairseq naming -> ours.
_FAIRSEQ_HUBERT_RULES = [
    (r"^feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv."),
    (r"^feature_extractor\.conv_layers\.0\.2\.", "feature_extractor.conv_layers.0.norm."),
    (r"^encoder\.pos_conv\.0\.", "encoder.pos_conv."),
]
_HUBERT_UNUSED = re.compile(r"^(mask_emb|label_embs_concat|masked_spec_embed|label_embs|encoder\.pos_conv\.0\.weight_[gv]$)")


def _rename(state: Dict[str, np.ndarray], rules) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in state.items():
        for pattern, repl in rules:
            k = re.sub(pattern, repl, k)
        out[k] = v
    return out


def _hubert_state_dict(ckpt) -> Dict[str, np.ndarray]:
    from rvc_mlx._checkpoint import fold_weight_norm, to_numpy_state_dict

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]  # fairseq checkpoint wrapper
    else:
        state = ckpt
    state = fold_weight_norm(to_numpy_state_dict(state))
    hf = any(k.startswith(("feature_projection.", "hubert.", "encoder.pos_conv_embed.")) for k in state)
    state = _rename(state, _HF_HUBERT_RULES if hf else _FAIRSEQ_HUBERT_RULES)
    return {k: v for k, v in state.items() if not _HUBERT_UNUSED.match(k)}


def _hubert_num_heads(ckpt, default: int = 12) -> int:
    """
    Attention head count isn't recoverable from the weights. Read it from a fairseq config when that config is a plain
    dict / namespace; otherwise fall back to HuBERT-Base's 12 (true for every checkpoint RVC ships).
    """
    if not isinstance(ckpt, dict):
        return default
    for key in ("cfg", "args"):
        node = ckpt.get(key)
        if isinstance(node, dict):
            node = node.get("model", node)
            value = node.get("encoder_attention_heads") if isinstance(node, dict) else None
        else:
            value = getattr(node, "encoder_attention_heads", None)
        if isinstance(value, int):
            return value
    return default


def convert_hubert_checkpoint(path: str, out_path: Optional[str] = None, dtype="float16") -> str:
    """
    Convert RVC's `hubert_base.pt` (fairseq) — or a Hugging Face ContentVec/HuBERT `pytorch_model.bin` /
    `model.safetensors` — to `hubert.safetensors`.
    """
    from rvc_mlx._checkpoint import load_torch_checkpoint, to_mlx_layout
    from rvc_mlx.hubert import HubertConfig, HubertModel
    from rvc_mlx.io import save_converted

    out_path = _default_out(path, out_path)
    if path.endswith(".safetensors"):
        ckpt = {k: np.array(v.astype(mx.float32)) for k, v in mx.load(path).items()}
        state = _hubert_state_dict(_NumpyDict(ckpt))
        num_heads = 12
    else:
        ckpt = load_torch_checkpoint(path)
        state = _hubert_state_dict(ckpt)
        num_heads = _hubert_num_heads(ckpt)

    layer_ids = {int(m.group(1)) for k in state if (m := re.match(r"^encoder\.layers\.(\d+)\.", k))}
    conv_ids = sorted(
        int(m.group(1)) for k in state if (m := re.match(r"^feature_extractor\.conv_layers\.(\d+)\.conv\.weight$", k))
    )
    default = HubertConfig()
    conv_layers = []
    for i in conv_ids:
        out_dim, _, kernel = state[f"feature_extractor.conv_layers.{i}.conv.weight"].shape
        stride = default.conv_layers[i][2] if i < len(default.conv_layers) else kernel
        conv_layers.append((int(out_dim), int(kernel), int(stride)))
    embed_dim, conv_pos_in, conv_pos = state["encoder.pos_conv.weight"].shape
    has_final_proj = "final_proj.weight" in state
    cfg = HubertConfig(
        conv_layers=conv_layers,
        embed_dim=int(embed_dim),
        ffn_dim=int(state["encoder.layers.0.fc1.weight"].shape[0]),
        num_heads=num_heads,
        num_layers=len(layer_ids),
        conv_pos=int(conv_pos),
        conv_pos_groups=int(embed_dim // conv_pos_in),
        final_dim=int(state["final_proj.weight"].shape[0]) if has_final_proj else 256,
    )
    model = HubertModel(cfg)
    if not has_final_proj:
        # Some ContentVec exports drop the projection head. It is only needed by v1 voices; keep the file loadable and
        # record the omission so v1 inference can fail loudly instead of producing garbage.
        state["final_proj.weight"] = np.zeros((cfg.final_dim, cfg.embed_dim), np.float32)
        state["final_proj.bias"] = np.zeros((cfg.final_dim,), np.float32)
    state = to_mlx_layout(model, state)
    model.load_weights([(k, mx.array(v)) for k, v in state.items()], strict=True)  # validates names and shapes
    save_converted(
        out_path,
        state,
        kind="hubert",
        metadata={"config": cfg.to_json(), "has_final_proj": "1" if has_final_proj else "0"},
        dtype=_resolve_dtype(dtype),
    )
    return out_path


class _NumpyDict(dict):
    """Marker type so `_hubert_state_dict` treats a safetensors dict as a bare state dict."""


def _json(obj) -> str:
    import json

    return json.dumps(obj)


# ---------------------------------------------------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------------------------------------------------


def synthesizer_weights_from_state_dict(state, cfg) -> Dict[str, np.ndarray]:
    """
    Turn an RVC synthesizer `state_dict` (torch tensors or numpy arrays) into MLX-layout weights for `Synthesizer(cfg)`,
    validating names and shapes. Training-only entries (`enc_q.*`) are dropped.
    """
    from rvc_mlx._checkpoint import fold_weight_norm, to_mlx_layout, to_numpy_state_dict
    from rvc_mlx.synthesizer import Synthesizer

    state = fold_weight_norm(to_numpy_state_dict(state))
    state = {k: v for k, v in state.items() if not k.startswith("enc_q.") and not k.endswith("num_batches_tracked")}
    model = Synthesizer(cfg)
    state = to_mlx_layout(model, state)
    model.load_weights([(k, mx.array(v)) for k, v in state.items()], strict=True)
    return state


def read_index_vectors(index_path: str) -> np.ndarray:
    """
    Read the retrieval feature bank of an RVC voice: a faiss `.index` (needs `faiss-cpu`) or the `total_fea.npy` /
    any `.npy` of shape (N, feature_dim) that RVC's training writes next to it.
    """
    if index_path.endswith(".npy"):
        return np.load(index_path).astype(np.float32)
    try:
        import faiss
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading a faiss .index needs `pip install faiss-cpu` (only at conversion time). "
            "Alternatively pass the voice's total_fea.npy."
        ) from e
    index = faiss.read_index(index_path)
    try:
        vectors = index.reconstruct_n(0, index.ntotal)
    except RuntimeError:
        faiss.extract_index_ivf(index).make_direct_map()
        vectors = index.reconstruct_n(0, index.ntotal)
    return np.asarray(vectors, dtype=np.float32)


def convert_voice_checkpoint(
    pth_path: str,
    out_path: Optional[str] = None,
    index_path: Optional[str] = None,
    name: Optional[str] = None,
    dtype="float16",
    max_index_vectors: Optional[int] = None,
) -> str:
    """
    Convert an RVC voice (`.pth` as exported by the RVC WebUI, plus optionally its `.index`) to one `.safetensors`.

    :param max_index_vectors: if the feature bank is larger, shrink it with k-means to this many centroids (RVC's own
        training does the same above 200k vectors). Useful for phones. `None` keeps every vector.
    """
    from rvc_mlx._checkpoint import load_torch_checkpoint
    from rvc_mlx.synthesizer import SynthConfig
    from rvc_mlx.voice import Voice

    ckpt = load_torch_checkpoint(pth_path)
    if not isinstance(ckpt, dict) or "weight" not in ckpt or "config" not in ckpt:
        raise ValueError(
            f"{pth_path!r} doesn't look like an RVC voice export (expected keys 'weight' and 'config'). "
            "Training checkpoints (G_*.pth) must first be exported with the RVC WebUI ('ckpt processing' tab)."
        )
    version = ckpt.get("version", "v1")
    f0 = bool(int(ckpt.get("f0", 1)))
    config = list(ckpt["config"])
    # RVC overwrites the speaker count with the real embedding table size when loading; do the same.
    config[-3] = int(ckpt["weight"]["emb_g.weight"].shape[0])
    cfg = SynthConfig.from_rvc(config, version=version, f0=f0)
    weights = synthesizer_weights_from_state_dict(ckpt["weight"], cfg)

    index = None
    if index_path is not None:
        index = read_index_vectors(index_path)
        if index.ndim != 2 or index.shape[1] != cfg.feature_dim:
            raise ValueError(
                f"Index vectors have shape {index.shape}, but a {version} voice needs (N, {cfg.feature_dim})."
            )
        if max_index_vectors is not None and index.shape[0] > max_index_vectors:
            from rvc_mlx.index import kmeans

            index = kmeans(index, max_index_vectors)

    voice = Voice.from_weights(
        cfg,
        {k: mx.array(v) for k, v in weights.items()},
        index=None if index is None else mx.array(index),
        name=name or os.path.splitext(os.path.basename(pth_path))[0],
        info=str(ckpt.get("info", "")),
    )
    return voice.save(_default_out(pth_path, out_path), dtype=_resolve_dtype(dtype))
