"""
HuBERT-style content encoder for RVC.

Ports the inference path of the ContentVec / HuBERT-base model RVC uses to extract per-frame content features from
raw audio. Output is the activations of one specific transformer layer (layer 9 for v1, layer 12 for v2), reshaped to
match the text-encoder's expected `(B, T, 768)` input.

Design choices:
- Channels-last MLX convention throughout (`(B, T, C)`). The PyTorch reference in `_torch_ref` mirrors the
  fairseq channels-first layout so paired-module tests can bridge weights.
- The transformer here is **pre-norm with GELU**, distinct from the synthesizer's post-norm transformer with
  optional relative-position embeddings. Q/K/V/O use `nn.Linear` (matching fairseq) rather than the synthesizer's
  1x1 `nn.Conv1d`.
- `pos_conv` is the convolutional positional encoding (Conv1d k=128 groups=16 wrapped in `weight_norm`). The released
  fairseq checkpoint stores it as `weight_g`/`weight_v` along axis 2; the converter fuses it back to plain `weight`
  before loading.
- Training-time pieces (`mask_emb`, `label_embs_concat`, the masking logic) are intentionally omitted. The inference
  path doesn't use them and they'd add a lot of plumbing.
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# HuBERT-base / ContentVec configuration (matches the released RVC `hubert_base.pt`).
HUBERT_BASE_CONV_LAYERS = (
    # (out_channels, kernel_size, stride)
    (512, 10, 5),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 2, 2),
    (512, 2, 2),
)
# Total downsampling factor: 5 * 2^6 = 320. At 16 kHz input, output frame rate is 50 Hz.
HUBERT_BASE_DOWNSAMPLE = 320

HUBERT_BASE_CONFIG = dict(
    conv_layers=HUBERT_BASE_CONV_LAYERS,
    extractor_mode="default",
    embed_dim=768,
    encoder_ffn_dim=3072,
    encoder_layers=12,
    encoder_attention_heads=12,
    pos_conv_kernel=128,
    pos_conv_groups=16,
    # ContentVec v2 (the variant RVC v2 uses) doesn't have `final_proj`; v1 does.
    has_final_proj=False,
)


class _SamePad(nn.Module):
    """
    Crop the right side of the time axis so a Conv1d with even kernel size preserves the input length. fairseq's
    `SamePad` does this with `if kernel_size % 2 == 0: x = x[..., :-1]` — for the channels-last MLX layout, "time
    last" becomes "time second-to-last", so the slice runs along axis -2.

    Used only by `pos_conv` (kernel_size=128 is even).
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.crop_one = kernel_size % 2 == 0

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C). Drop the last time step if the conv produced one extra.
        if self.crop_one:
            return x[:, :-1, :]
        return x


class FeatureExtractor(nn.Module):
    """
    Convolutional feature extractor: 7 strided 1D convolutions that turn raw 16 kHz audio into 512-dim features at
    50 Hz frame rate.

    Mode "default" (the only mode RVC uses): the first conv layer has a GroupNorm with `num_groups == num_channels`
    (equivalent to LayerNorm across time per channel); subsequent layers have no normalization. Activation is GELU
    after every layer.
    """

    def __init__(
        self,
        conv_layers,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()
        if mode != "default":
            raise NotImplementedError(
                f"FeatureExtractor mode {mode!r} is not implemented. RVC's hubert_base.pt uses 'default'."
            )
        self.conv_layers_config = tuple(conv_layers)
        self.convs: list = []
        # `norms[i]` is None unless layer i carries a GroupNorm. Kept parallel to `convs` so the per-layer loop is
        # straightforward.
        self.norms: list = []
        in_d = 1
        for i, (out_d, k, s) in enumerate(self.conv_layers_config):
            self.convs.append(
                nn.Conv1d(in_d, out_d, kernel_size=k, stride=s, bias=conv_bias)
            )
            if i == 0:
                # num_groups == out_d: each channel is its own group, equivalent to per-channel layer norm.
                self.norms.append(
                    nn.GroupNorm(num_groups=out_d, dims=out_d, affine=True, pytorch_compatible=True)
                )
            else:
                self.norms.append(None)
            in_d = out_d

    def __call__(self, audio: mx.array) -> mx.array:
        # audio: (B, T_audio) -> add channel axis to (B, T_audio, 1).
        x = mx.expand_dims(audio, -1)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = nn.gelu(x)
        return x  # (B, T_out, 512), T_out ≈ T_audio // 320


class HubertMultiHeadAttention(nn.Module):
    """
    Standard scaled-dot-product self-attention used inside HuBERT's transformer encoder. Q/K/V/O are `nn.Linear`
    projections (matching fairseq's `MultiheadAttention` for the `batch_first=True` case). No relative-position
    embeddings — HuBERT uses a separate convolutional positional encoding (`pos_conv`) instead.

    Input/output: `(B, T, embed_dim)`.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        key_padding_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # x: (B, T, embed_dim). For RVC's inference path key/query/value all come from the same x, so we take a
        # single argument here.
        B, T, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )  # (B, H, T, D)
        k = (
            self.k_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        scores = mx.matmul(q * self.scaling, mx.swapaxes(k, -2, -1))  # (B, H, T, T)

        if key_padding_mask is not None:
            # key_padding_mask: (B, T_k), True at masked (invalid) positions.
            mask = mx.expand_dims(mx.expand_dims(key_padding_mask, 1), 1)  # (B, 1, 1, T_k)
            scores = mx.where(mask, mx.array(-1e9, dtype=scores.dtype), scores)

        attn = mx.softmax(scores, axis=-1)
        attn = self.dropout(attn)
        out = mx.matmul(attn, v)  # (B, H, T, D)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, T, self.embed_dim)
        out = self.out_proj(out)
        return out


class TransformerSentenceEncoderLayer(nn.Module):
    """
    A single transformer block from HuBERT's encoder. **Pre-norm** with GELU activation in the FFN:

        x = x + dropout(self_attn(norm(x)))
        x = x + dropout(fc2(dropout(gelu(fc1(norm(x))))))

    Module names (`self_attn`, `self_attn_layer_norm`, `fc1`, `fc2`, `final_layer_norm`) match fairseq exactly so the
    state_dict converter can map keys one-to-one.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = HubertMultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def __call__(
        self, x: mx.array, key_padding_mask: Optional[mx.array] = None
    ) -> mx.array:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = residual + x
        return x


class PositionalConv(nn.Module):
    """
    Convolutional positional encoding from HuBERT's transformer encoder. A grouped 1D conv (kernel=128, groups=16)
    with weight_norm + GELU. In the released fairseq checkpoint this is `pos_conv.0.{weight_g,weight_v,bias}` (the
    `0` is the index into the `nn.Sequential(conv, SamePad, GELU)`).

    Our MLX side stores it as a plain Conv1d; the checkpoint converter fuses `weight_g`/`weight_v` (with `dim=2`, the
    kernel-position axis) into a single `weight`.
    """

    def __init__(self, embed_dim: int, kernel_size: int = 128, groups: int = 16):
        super().__init__()
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.same_pad = _SamePad(kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, embed_dim). Conv with padding=k//2 makes output length T+1 for even k; SamePad crops back to T.
        out = self.conv(x)
        out = self.same_pad(out)
        return nn.gelu(out)


class HubertTransformerEncoder(nn.Module):
    """
    HuBERT's transformer encoder: positional conv + post-pos LayerNorm + a stack of
    `TransformerSentenceEncoderLayer`s.

    `__call__` returns the activations after each transformer layer in `layer_results` so callers can extract from
    any depth (RVC v1 reads from layer 9, v2 from layer 12).
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_layers: int,
        num_heads: int,
        pos_conv_kernel: int = 128,
        pos_conv_groups: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.pos_conv = PositionalConv(embed_dim, pos_conv_kernel, pos_conv_groups)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layers = [
            TransformerSentenceEncoderLayer(embed_dim, ffn_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        padding_mask: Optional[mx.array] = None,
        output_layer: Optional[int] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        # x: (B, T, embed_dim). The positional encoding is added back to x; then we run through the transformer
        # stack, optionally stopping early at `output_layer` (1-indexed).
        x_pos = self.pos_conv(x)
        x = x + x_pos
        x = self.layer_norm(x)

        layer_results: List[mx.array] = []
        for i, layer in enumerate(self.layers):
            x = layer(x, key_padding_mask=padding_mask)
            layer_results.append(x)
            if output_layer is not None and (i + 1) >= output_layer:
                break
        return x, layer_results


class HubertModel(nn.Module):
    """
    The full HuBERT / ContentVec content encoder used by RVC. Inference pipeline:

      audio (B, T_audio)
        -> FeatureExtractor    -> (B, T_out, 512)
        -> layer_norm          -> (B, T_out, 512)      (LayerNorm on the feature dim)
        -> post_extract_proj   -> (B, T_out, 768)
        -> HubertTransformerEncoder(...output_layer=k) -> (B, T_out, 768)
        -> [optional] final_proj  (only present in v1 / non-ContentVec checkpoints)

    The `extract_features` method exposes the result at a chosen transformer layer, matching the contract of
    `fairseq.HubertModel.extract_features`.
    """

    def __init__(
        self,
        conv_layers=HUBERT_BASE_CONV_LAYERS,
        extractor_mode: str = "default",
        embed_dim: int = 768,
        encoder_ffn_dim: int = 3072,
        encoder_layers: int = 12,
        encoder_attention_heads: int = 12,
        pos_conv_kernel: int = 128,
        pos_conv_groups: int = 16,
        has_final_proj: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.has_final_proj = has_final_proj
        self.downsample_factor = math.prod(s for _, _, s in conv_layers)

        self.feature_extractor = FeatureExtractor(conv_layers, mode=extractor_mode)
        feature_dim = conv_layers[-1][0]  # 512 for HuBERT-base
        # Post-feature-extractor LayerNorm. In fairseq this is stored as `layer_norm.*` (top-level), distinct from
        # the encoder's internal `encoder.layer_norm.*`.
        self.layer_norm = nn.LayerNorm(feature_dim)
        # Projection from feature_dim (512) to encoder embed_dim (768). Always present for HuBERT-base.
        self.post_extract_proj = nn.Linear(feature_dim, embed_dim)
        self.encoder = HubertTransformerEncoder(
            embed_dim=embed_dim,
            ffn_dim=encoder_ffn_dim,
            num_layers=encoder_layers,
            num_heads=encoder_attention_heads,
            pos_conv_kernel=pos_conv_kernel,
            pos_conv_groups=pos_conv_groups,
        )
        if has_final_proj:
            self.final_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.final_proj = None

    @classmethod
    def from_pretrained(cls, path: str) -> "HubertModel":
        """
        Build a `HubertModel` with weights loaded from disk.

        Accepts either an MLX-native `.safetensors` (with sibling `*.config.json`) or the released fairseq
        `hubert_base.pt` / ContentVec `.pt` checkpoint. For the `.pt` path the conversion runs on first use,
        producing both files; subsequent loads skip torch entirely. The `has_final_proj` flag is auto-detected
        from the checkpoint contents (true for HuBERT-base v1, false for ContentVec v2).
        """
        from rvc_mlx.convert import ensure_hubert_safetensors

        safetensors_path, config = ensure_hubert_safetensors(path)
        model = cls(**config)
        model.load_weights(safetensors_path)
        model.eval()
        return model

    def extract_features(
        self,
        audio: mx.array,
        padding_mask: Optional[mx.array] = None,
        output_layer: Optional[int] = None,
    ) -> mx.array:
        """
        :param audio: raw 16 kHz audio, shape `(B, T_audio)`.
        :param padding_mask: optional `(B, T_out)` bool mask where True marks invalid frames. If `None`, no masking
            is applied (the typical RVC inference case).
        :param output_layer: 1-indexed transformer layer to extract from. `None` returns the output of the final
            layer. RVC v1 uses `output_layer=9`, v2 uses `output_layer=12`.
        :returns: features of shape `(B, T_out, embed_dim)`.
        """
        x = self.feature_extractor(audio)  # (B, T_out, 512)
        x = self.layer_norm(x)
        x = self.post_extract_proj(x)  # (B, T_out, 768)
        x, layer_results = self.encoder(x, padding_mask=padding_mask, output_layer=output_layer)
        if output_layer is None or output_layer == self.encoder.num_layers:
            out = x
        else:
            out = layer_results[output_layer - 1]
        if self.has_final_proj and self.final_proj is not None:
            out = self.final_proj(out)
        return out
