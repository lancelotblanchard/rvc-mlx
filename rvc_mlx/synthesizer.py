"""
RVC voice-conversion synthesizer modules.

Mirrors the inference path of `SynthesizerTrnMs768NSFsid` from the RVC reference implementation. The reference uses
PyTorch channels-first conventions throughout (Conv1d takes `(B, C, T)`); MLX is channels-last natively (`(B, T, C)`).
To keep call-sites readable and the operations idiomatic, the MLX modules in this file work in channels-last natively
and the test harness / weight-copy bridge handles the transpose at the boundary with the PyTorch reference.

Module layout follows the original RVC source so that paired-module tests (one MLX impl + one PyTorch ref) compare
faithfully and a future safetensors converter can mirror the structure.

This file currently implements the transformer-encoder building blocks (LayerNorm, FFN, MultiHeadAttention, Encoder).
Higher-level pieces (TextEncoder768, GeneratorNSF, ResidualCouplingBlock, SynthesizerTrnMs768NSFsid) will land in
follow-up commits.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from rvc_mlx.utils import pad_constant, sequence_mask


class LayerNorm(nn.Module):
    """
    Per-channel layer normalization used throughout RVC. The PyTorch reference computes
    `F.layer_norm(x.transpose(1, -1), (channels,), gamma, beta, eps)` and transposes back, so the normalization runs
    over the channel axis. In MLX channels-last convention the channel axis is already last, so we can normalize over
    the last dim directly without any transpose.

    Parameter names (`gamma`, `beta`) mirror the RVC reference so the cross-framework copy helper is unambiguous.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = mx.ones((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (..., channels)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        return x * self.gamma + self.beta


def _same_padding(x: mx.array, kernel_size: int) -> mx.array:
    """
    Pad along the time axis (second-to-last in MLX channels-last) so a Conv1d of `kernel_size` preserves the length.
    Mirrors the RVC reference's `_same_padding`, which splits `kernel_size - 1` total padding as
    `(kernel_size - 1) // 2` on the left and `kernel_size // 2` on the right (asymmetric for even kernels).
    """
    if kernel_size == 1:
        return x
    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size // 2
    # In MLX channels-last (B, T, C), padding T is the second-to-last dim. pad_constant follows the PyTorch convention
    # of last-dim-first: (last_l, last_r, second-to-last_l, second-to-last_r, ...).
    return pad_constant(x, (0, 0, pad_l, pad_r), value=0.0)


def _causal_padding(x: mx.array, kernel_size: int) -> mx.array:
    """Left-only padding along time so the receptive field never reads future frames. Matches RVC `_causal_padding`."""
    if kernel_size == 1:
        return x
    pad_l = kernel_size - 1
    return pad_constant(x, (0, 0, pad_l, 0), value=0.0)


class FFN(nn.Module):
    """
    Position-wise feed-forward network used inside the transformer encoder. Two Conv1d layers (acting as point-wise
    Linear when `kernel_size=1`, otherwise convolving across time) with an activation in between.

    Activation is either ReLU (default) or a sigmoid-linear "GELU approximation" `x * sigmoid(1.702 * x)` selected by
    `activation="gelu"`, exactly matching the RVC reference.

    Channels-last MLX convention: input/output are `(B, T, C)`, mask is `(B, T, 1)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: Optional[str] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.is_gelu = activation == "gelu"

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def _pad(self, x: mx.array) -> mx.array:
        if self.causal:
            return _causal_padding(x, self.kernel_size)
        return _same_padding(x, self.kernel_size)

    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        # x: (B, T, C_in), x_mask: (B, T, 1)
        x = self.conv_1(self._pad(x * x_mask))
        if self.is_gelu:
            x = x * mx.sigmoid(1.702 * x)
        else:
            x = nn.relu(x)
        x = self.drop(x)
        x = self.conv_2(self._pad(x * x_mask))
        return x * x_mask


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional relative-position embeddings, matching the RVC reference.

    Channels-last MLX convention: input/output are `(B, T, C)`. Q/K/V projections are 1x1 Conv1d (equivalent to per-time
    Linear). Relative-position bias is applied to the attention scores when `window_size` is set.

    Mask convention: `attn_mask` has shape `(B, 1, T_q, T_k)` matching the head-broadcasted layout used in the original
    reference. Zero entries are treated as "mask out" (set to a large negative number before softmax).
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        heads_share: bool = True,
        block_length: Optional[int] = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_o = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = mx.random.normal((n_heads_rel, window_size * 2 + 1, self.k_channels)) * rel_stddev
            self.emb_rel_v = mx.random.normal((n_heads_rel, window_size * 2 + 1, self.k_channels)) * rel_stddev

    def __call__(
        self, x: mx.array, c: mx.array, attn_mask: Optional[mx.array] = None
    ) -> mx.array:
        # x, c: (B, T, C). For self-attention, x is c.
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x = self._attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def _attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # query/key/value: (B, T, C)
        b, t_t, _ = query.shape
        t_s = key.shape[1]
        # Reshape (B, T, C) -> (B, T, n_heads, k_channels) -> (B, n_heads, T, k_channels)
        query = mx.transpose(query.reshape(b, t_t, self.n_heads, self.k_channels), (0, 2, 1, 3))
        key = mx.transpose(key.reshape(b, t_s, self.n_heads, self.k_channels), (0, 2, 1, 3))
        value = mx.transpose(value.reshape(b, t_s, self.n_heads, self.k_channels), (0, 2, 1, 3))

        scale = 1.0 / math.sqrt(self.k_channels)
        # scores: (B, H, T_q, T_k)
        scores = mx.matmul(query * scale, mx.swapaxes(key, -2, -1))

        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query * scale, key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).astype(scores.dtype)
        if mask is not None:
            scores = mx.where(mask == 0, -1e4, scores)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                # Lower-triangular AND upper-triangular constraint: keep band of width block_length on each side.
                idx = mx.arange(t_s)
                # block_mask[i, j] = 1 if |i - j| <= block_length, else 0.
                diff = mx.expand_dims(idx, 0) - mx.expand_dims(idx, 1)
                band = (mx.abs(diff) <= self.block_length).astype(scores.dtype)
                # Broadcast to (1, 1, T, T) for compatibility with scores (B, H, T, T).
                band = mx.expand_dims(mx.expand_dims(band, 0), 0)
                scores = mx.where(band == 0, -1e4, scores)
        p_attn = mx.softmax(scores, axis=-1)  # (B, H, T_q, T_k)
        p_attn = self.drop(p_attn)
        # output: (B, H, T_q, k_channels)
        output = mx.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        # Back to (B, T_q, n_heads, k_channels) -> (B, T_q, C)
        output = mx.transpose(output, (0, 2, 1, 3)).reshape(b, t_t, self.channels)
        return output

    # --- Relative-position helpers (line-for-line ports of the RVC reference, adapted for channels-last). -----------

    @staticmethod
    def _matmul_with_relative_values(x: mx.array, y: mx.array) -> mx.array:
        """
        x: (B, H, L, M)
        y: (H_or_1, M, D)
        ret: (B, H, L, D)
        """
        return mx.matmul(x, mx.expand_dims(y, 0))

    @staticmethod
    def _matmul_with_relative_keys(x: mx.array, y: mx.array) -> mx.array:
        """
        x: (B, H, L, D)
        y: (H_or_1, M, D)
        ret: (B, H, L, M)
        """
        return mx.matmul(x, mx.swapaxes(mx.expand_dims(y, 0), -2, -1))

    def _get_relative_embeddings(self, relative_embeddings: mx.array, length: int) -> mx.array:
        # relative_embeddings: (H_or_1, 2 * window_size + 1, k_channels)
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            # Pad middle dim (axis -2) by `pad_length` on each side. In pad_constant's last-dim-first convention:
            # last dim (k_channels): (0, 0); middle dim (relative pos): (pad_length, pad_length); first dim: (0, 0).
            padded = pad_constant(
                relative_embeddings, (0, 0, pad_length, pad_length, 0, 0), value=0.0
            )
        else:
            padded = relative_embeddings
        return padded[:, slice_start:slice_end]

    @staticmethod
    def _relative_position_to_absolute_position(x: mx.array) -> mx.array:
        """
        x: (B, H, L, 2L - 1) -> (B, H, L, L)

        The trick: pad-and-reshape so that absolute positions fall on the diagonal of a (L+1, 2L-1) view, then slice
        out the relevant (L, L) submatrix. Line-for-line port of the RVC reference.
        """
        batch, heads, length, _ = x.shape
        # Pad last dim right by 1: (B, H, L, 2L)
        x = pad_constant(x, (0, 1, 0, 0, 0, 0, 0, 0), value=0.0)
        x_flat = x.reshape(batch, heads, length * 2 * length)
        # Pad the now-flat last dim right by (length - 1)
        x_flat = pad_constant(x_flat, (0, length - 1, 0, 0, 0, 0), value=0.0)
        x_final = x_flat.reshape(batch, heads, length + 1, 2 * length - 1)[
            :, :, :length, length - 1 :
        ]
        return x_final

    @staticmethod
    def _absolute_position_to_relative_position(x: mx.array) -> mx.array:
        """
        x: (B, H, L, L) -> (B, H, L, 2L - 1)

        Inverse of `_relative_position_to_absolute_position` via the same pad-and-reshape trick.
        """
        batch, heads, length, _ = x.shape
        x = pad_constant(x, (0, length - 1, 0, 0, 0, 0, 0, 0), value=0.0)
        x_flat = x.reshape(batch, heads, length * length + length * (length - 1))
        x_flat = pad_constant(x_flat, (length, 0, 0, 0, 0, 0), value=0.0)
        x_final = x_flat.reshape(batch, heads, length, 2 * length)[:, :, :, 1:]
        return x_final

    @staticmethod
    def _attention_bias_proximal(length: int) -> mx.array:
        """
        Bias for self-attention to prefer close positions. Returns shape (1, 1, length, length).

        Note: the RVC reference computes this in float32 even when the surrounding tensors are half. We do the same:
        the caller casts the result to scores.dtype at the call site.
        """
        r = mx.arange(length, dtype=mx.float32)
        diff = mx.expand_dims(r, 0) - mx.expand_dims(r, 1)
        return mx.expand_dims(mx.expand_dims(-mx.log1p(mx.abs(diff)), 0), 0)


class Encoder(nn.Module):
    """
    Transformer encoder used inside `TextEncoder768`. A stack of `n_layers` (self-attention + LayerNorm + residual)
    followed by (FFN + LayerNorm + residual) blocks. Pre-norm or post-norm? The RVC reference is **post-norm**:
    `x = norm(x + sublayer(x))`.

    Channels-last MLX convention: input/output are `(B, T, C)`, mask is `(B, T, 1)`. The attention mask is computed
    inside this module from `x_mask` so callers don't need to construct it.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = int(n_layers)
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = []
        self.norm_layers_1 = []
        self.ffn_layers = []
        self.norm_layers_2 = []
        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        # x: (B, T, C), x_mask: (B, T, 1)
        # Build attn_mask of shape (B, 1, T, T) from x_mask: 1 iff both query and key positions are valid.
        m = mx.squeeze(x_mask, -1)  # (B, T)
        attn_mask = mx.expand_dims(
            mx.expand_dims(m, -1) * mx.expand_dims(m, -2), 1
        )  # (B, 1, T, T)
        x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            y = attn(x, x, attn_mask)
            y = self.drop(y)
            x = norm1(x + y)

            y = ffn(x, x_mask)
            y = self.drop(y)
            x = norm2(x + y)
        x = x * x_mask
        return x


class TextEncoder768(nn.Module):
    """
    The text encoder used by `SynthesizerTrnMs768NSFsid`. Maps 768-dim per-frame phone features (typically HuBERT-like
    content embeddings, hence the `768` in the name) plus an integer pitch class per frame to a posterior parameterized
    by `(m, logs)` of shape `(B, T, out_channels)`. The transformer encoder runs in the middle.

    Channels-last MLX convention: `phone` is `(B, T, 768)`, `pitch` is `(B, T)` integers in `[0, 255]` (or None for the
    "no pitch" variant), and `lengths` is `(B,)`. The returned `m`, `logs` are `(B, T, out_channels)` and `x_mask` is
    `(B, T, 1)`. The PyTorch reference uses channels-first; the test bridge transposes at the boundary.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = nn.Linear(768, hidden_channels)
        # The reference uses `LeakyReLU(0.1, inplace=True)`; inplace is a no-op semantics-wise for the test.
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        # 1x1 Conv1d projecting hidden_channels -> 2 * out_channels (split into mean and log-std).
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)

    def __call__(
        self,
        phone: mx.array,
        pitch: Optional[mx.array],
        lengths: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        # phone: (B, T, 768); pitch: (B, T) or None; lengths: (B,)
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        # Build x_mask of shape (B, T, 1) where T is the second-to-last axis of x.
        T = x.shape[1]
        mask = sequence_mask(lengths, T).astype(x.dtype)  # (B, T)
        x_mask = mx.expand_dims(mask, -1)  # (B, T, 1)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask  # (B, T, 2 * out_channels)
        # Split along the channel axis. RVC's `torch.split(stats, out_channels, dim=1)` on channels-first becomes a
        # last-axis split here.
        m = stats[..., : self.out_channels]
        logs = stats[..., self.out_channels :]
        return m, logs, x_mask
