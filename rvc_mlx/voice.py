"""
A converted RVC voice: synthesizer weights + architecture config + optional retrieval feature bank, stored together in
one `.safetensors` file (see `rvc_mlx.io`), plus voice blending.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import mlx.core as mx
from mlx.utils import tree_flatten

from rvc_mlx.io import load_converted, save_converted
from rvc_mlx.synthesizer import SynthConfig, Synthesizer

INDEX_KEY = "index.vectors"


class Voice:
    def __init__(
        self,
        cfg: SynthConfig,
        model: Synthesizer,
        index: Optional[mx.array] = None,
        name: str = "voice",
        info: str = "",
        metadata: Optional[Dict[str, str]] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.index = index
        self.name = name
        self.info = info
        self.metadata = dict(metadata or {})

    # ------------------------------------------------------------------------------------------------------- props
    @property
    def sample_rate(self) -> int:
        return self.cfg.sr

    @property
    def version(self) -> str:
        return self.cfg.version

    @property
    def has_f0(self) -> bool:
        return self.cfg.f0

    @property
    def num_speakers(self) -> int:
        return self.cfg.spk_embed_dim

    def __repr__(self) -> str:
        idx = "no index" if self.index is None else f"index {self.index.shape[0]}x{self.index.shape[1]}"
        return f"Voice({self.name!r}, {self.version}, {self.sample_rate} Hz, f0={self.has_f0}, {idx})"

    # ---------------------------------------------------------------------------------------------------------- io
    @classmethod
    def from_weights(cls, cfg: SynthConfig, weights: Dict[str, mx.array], index=None, name="voice", info="") -> "Voice":
        model = Synthesizer(cfg)
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        return cls(cfg, model, index=index, name=name, info=info)

    @classmethod
    def load(cls, path: str, dtype: mx.Dtype = mx.float32) -> "Voice":
        weights, meta = load_converted(path, expected_kind="voice", dtype=dtype)
        cfg = SynthConfig.from_json(meta["config"])
        index = weights.pop(INDEX_KEY, None)
        voice = cls.from_weights(cfg, weights, index=index, name=meta.get("name", "voice"), info=meta.get("info", ""))
        voice.metadata = meta
        return voice

    def weights(self) -> Dict[str, mx.array]:
        return dict(tree_flatten(self.model.parameters()))

    def save(self, path: str, dtype: Optional[mx.Dtype] = mx.float16) -> str:
        arrays = self.weights()
        if self.index is not None:
            arrays[INDEX_KEY] = self.index
        meta = {
            "name": self.name,
            "info": self.info,
            "version": self.cfg.version,
            "f0": "1" if self.cfg.f0 else "0",
            "sample_rate": str(self.cfg.sr),
            "config": self.cfg.to_json(),
            "index_size": "0" if self.index is None else str(self.index.shape[0]),
        }
        for k in ("merged_from",):
            if k in self.metadata:
                meta[k] = self.metadata[k]
        return save_converted(path, arrays, kind="voice", metadata=meta, dtype=dtype)


def _arch(cfg: SynthConfig) -> dict:
    d = json.loads(cfg.to_json())
    d.pop("spk_embed_dim")
    return d


def check_compatible(voices: Sequence[Voice]) -> None:
    """Raise if the voices can't be blended (different version / sample rate / layer sizes)."""
    ref = _arch(voices[0].cfg)
    for v in voices[1:]:
        if _arch(v.cfg) != ref:
            diff = {k: (ref[k], _arch(v.cfg)[k]) for k in ref if ref[k] != _arch(v.cfg)[k]}
            raise ValueError(f"Voices {voices[0].name!r} and {v.name!r} have different architectures: {diff}")


def blend_weights(voices: Sequence[Voice], weights: Sequence[float]) -> Dict[str, mx.array]:
    """
    Weighted average of the synthesizer parameters (weights are normalised to sum to 1), as RVC's "ckpt merge" does.
    Speaker tables of different sizes are truncated to the smallest one.
    """
    check_compatible(voices)
    if len(voices) != len(weights) or not voices:
        raise ValueError("Need one weight per voice")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Blend weights must sum to a positive number")
    ws = [w / total for w in weights]
    params = [v.weights() for v in voices]
    out = {}
    for key in params[0]:
        arrays = [p[key] for p in params]
        if key == "emb_g.weight":
            n = min(a.shape[0] for a in arrays)
            arrays = [a[:n] for a in arrays]
        acc = arrays[0].astype(mx.float32) * ws[0]
        for a, w in zip(arrays[1:], ws[1:]):
            acc = acc + a.astype(mx.float32) * w
        out[key] = acc.astype(arrays[0].dtype)
    return out


def blend_voices(voices: Sequence[Voice], weights: Sequence[float], name: Optional[str] = None) -> Voice:
    """
    Create a new voice whose weights interpolate the given ones. The retrieval banks are concatenated, so retrieval
    can land on frames of any of the source voices.
    """
    blended = blend_weights(voices, weights)
    cfg = SynthConfig.from_json(voices[0].cfg.to_json())
    cfg.spk_embed_dim = int(blended["emb_g.weight"].shape[0])
    banks: List[mx.array] = [v.index for v in voices if v.index is not None]
    index = mx.concatenate(banks, axis=0) if banks else None
    total = float(sum(weights))
    recipe = ", ".join(f"{v.name}:{w / total:.3f}" for v, w in zip(voices, weights))
    voice = Voice.from_weights(cfg, blended, index=index, name=name or " + ".join(v.name for v in voices))
    voice.metadata["merged_from"] = recipe
    return voice
