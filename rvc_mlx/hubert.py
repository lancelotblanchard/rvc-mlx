"""
HuBERT / ContentVec content encoder used by RVC, in MLX.

RVC conditions its synthesizer on frame-level "content" features extracted by the `hubert_base.pt` checkpoint (a
fairseq HuBERT-Base model, also distributed as ContentVec). v1 voices read the 9th transformer layer and project it to
256 dims with `final_proj`; v2 voices read the 12th (last) layer at its native 768 dims.

Layout follows MLX conventions: audio is `(B, L)`, activations are channels-last `(B, T, C)`. Parameter names mirror
fairseq's where the module tree allows it; `rvc_mlx.convert` owns the exact key mapping.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class HubertConfig:
    conv_layers: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    )
    embed_dim: int = 768
    ffn_dim: int = 3072
    num_heads: int = 12
    num_layers: int = 12
    conv_pos: int = 128
    conv_pos_groups: int = 16
    final_dim: int = 256

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "HubertConfig":
        d = json.loads(s)
        d["conv_layers"] = [tuple(c) for c in d["conv_layers"]]
        return cls(**d)


class ChannelGroupNorm(nn.Module):
    """`GroupNorm(C, C)` on channels-last input: each channel is normalised over time independently (in fp32)."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        return (x * self.weight + self.bias).astype(dtype)


class FeatureExtractorLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, stride: int, group_norm: bool):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel, stride=stride, bias=False)
        if group_norm:
            self.norm = ChannelGroupNorm(out_dim)
        self._group_norm = group_norm

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self._group_norm:
            x = self.norm(x)
        return nn.gelu(x)


class FeatureExtractor(nn.Module):
    def __init__(self, conv_layers):
        super().__init__()
        self.conv_layers = []
        in_dim = 1
        for i, (dim, k, stride) in enumerate(conv_layers):
            self.conv_layers.append(FeatureExtractorLayer(in_dim, dim, k, stride, group_norm=(i == 0)))
            in_dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.conv_layers:
            x = layer(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dims // num_heads) ** -0.5
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        H = self.num_heads

        def heads(t):
            return t.reshape(B, T, H, C // H).transpose(0, 2, 1, 3)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, T, C))


class EncoderLayer(nn.Module):
    """Post-LayerNorm transformer block (fairseq `TransformerSentenceEncoderLayer`, `layer_norm_first=False`)."""

    def __init__(self, dims: int, ffn_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = SelfAttention(dims, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(dims, eps=1e-5)
        self.fc1 = nn.Linear(dims, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dims)
        self.final_layer_norm = nn.LayerNorm(dims, eps=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.self_attn_layer_norm(x + self.self_attn(x))
        x = self.final_layer_norm(x + self.fc2(nn.gelu(self.fc1(x))))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: HubertConfig):
        super().__init__()
        # Grouped positional convolution. fairseq wraps it in weight-norm (dim=2) + SamePad + GELU; the converter folds
        # the weight norm, and `__call__` reproduces SamePad by dropping the extra trailing frame.
        self.pos_conv = nn.Conv1d(
            cfg.embed_dim,
            cfg.embed_dim,
            cfg.conv_pos,
            padding=cfg.conv_pos // 2,
            groups=cfg.conv_pos_groups,
        )
        self._same_pad_remove = 1 if cfg.conv_pos % 2 == 0 else 0
        self.layer_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-5)
        self.layers = [EncoderLayer(cfg.embed_dim, cfg.ffn_dim, cfg.num_heads) for _ in range(cfg.num_layers)]

    def __call__(self, x: mx.array, num_layers: Optional[int] = None) -> mx.array:
        x_conv = self.pos_conv(x)
        if self._same_pad_remove:
            x_conv = x_conv[:, : -self._same_pad_remove, :]
        x = self.layer_norm(x + nn.gelu(x_conv))
        n = len(self.layers) if num_layers is None else num_layers
        for layer in self.layers[:n]:
            x = layer(x)
        return x


class HubertModel(nn.Module):
    def __init__(self, cfg: Optional[HubertConfig] = None):
        super().__init__()
        cfg = cfg or HubertConfig()
        self.cfg = cfg
        last_conv_dim = cfg.conv_layers[-1][0]
        self.feature_extractor = FeatureExtractor(cfg.conv_layers)
        self.layer_norm = nn.LayerNorm(last_conv_dim, eps=1e-5)
        self.post_extract_proj = nn.Linear(last_conv_dim, cfg.embed_dim)
        self.encoder = Encoder(cfg)
        self.final_proj = nn.Linear(cfg.embed_dim, cfg.final_dim)

    def extract_features(self, source: mx.array, output_layer: Optional[int] = None) -> mx.array:
        """
        Run the encoder up to (and including) transformer layer `output_layer` (1-based, fairseq convention).

        :param source: raw 16 kHz audio, `(B, L)` or `(L,)`.
        :returns: `(B, T, embed_dim)` features at 50 frames per second.
        """
        if source.ndim == 1:
            source = source[None]
        x = self.feature_extractor(source[..., None])
        x = self.post_extract_proj(self.layer_norm(x))
        return self.encoder(x, num_layers=output_layer)

    def __call__(self, source: mx.array, version: str = "v2") -> mx.array:
        """RVC content features: layer 9 + `final_proj` (256-d) for v1 voices, layer 12 (768-d) for v2."""
        if version == "v1":
            return self.final_proj(self.extract_features(source, output_layer=9))
        if version == "v2":
            return self.extract_features(source, output_layer=12)
        raise ValueError(f"Unknown RVC version {version!r}; expected 'v1' or 'v2'.")

    @classmethod
    def from_pretrained(cls, path: str, dtype: mx.Dtype = mx.float32) -> "HubertModel":
        """Load a converted `hubert.safetensors` (see `rvc_mlx.convert.convert_hubert_checkpoint`)."""
        from rvc_mlx.io import load_converted

        weights, meta = load_converted(path, expected_kind="hubert", dtype=dtype)
        cfg = HubertConfig.from_json(meta["config"]) if "config" in meta else HubertConfig()
        model = cls(cfg)
        model.load_weights(list(weights.items()))
        model.eval()
        return model
