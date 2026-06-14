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

from rvc_mlx.utils import (
    interpolate_linear_axis,
    interpolate_nearest_axis,
    pad_constant,
    sequence_mask,
)


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


# ----------------------------------------------------------------------------------------------------------------------
# NSF (Neural Source-Filter) source modules.
#
# The generator uses harmonic-plus-noise modeling: SineGen produces a bank of sine waves at multiples of f0 (with
# random noise in unvoiced regions), SourceModuleHnNSF mixes them down to a single excitation signal that the
# upsampling Conv stack consumes alongside the upsampled latents.
#
# SineGen's PyTorch reference is intrinsically random (it samples a per-harmonic initial phase and Gaussian noise
# for unvoiced regions). For cross-framework testing we expose those random tensors as optional `rand_ini` and
# `noise_raw` keyword arguments; when omitted, they are generated internally as in the reference.
# ----------------------------------------------------------------------------------------------------------------------


class SineGen(nn.Module):
    """
    Harmonic sine-wave generator matching the RVC reference.

    Inputs:
      * `f0`: per-frame fundamental frequency in Hz, shape `(B, T)`. Zero or sub-`voiced_threshold` entries are treated
        as unvoiced.
      * `upp`: integer upsampling factor from frame rate to audio rate (typically `prod(upsample_rates)`).
      * `rand_ini` (optional): `(B, dim)` per-harmonic initial-phase random values in `[0, 1)`. If `None`, sampled
        internally.
      * `noise_raw` (optional): unit-variance random tensor of shape `(B, T * upp, dim)` used to fill unvoiced regions.
        If `None`, sampled internally.

    Outputs `(sine_waves, uv, noise)`, each of shape `(B, T * upp, dim)` where `dim = harmonic_num + 1`.
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
        flag_for_pulse: bool = False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        # `flag_for_pulse` is part of the reference signature; the inference path never sets it to True. Stored
        # verbatim so downstream consumers can read it but otherwise unused.
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0: mx.array) -> mx.array:
        # f0: any shape. Returns a same-shape float array, 1 where f0 > threshold and 0 elsewhere.
        return (f0 > self.voiced_threshold).astype(f0.dtype)

    def __call__(
        self,
        f0: mx.array,
        upp: int,
        rand_ini: Optional[mx.array] = None,
        noise_raw: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        # f0: (B, T). The reference does `f0[:, None].transpose(1, 2)` which produces (B, T, 1). Channels-last MLX
        # version: just expand_dims on the last axis.
        f0_e = mx.expand_dims(f0, -1)  # (B, T, 1)
        B, T, _ = f0_e.shape

        # Build f0_buf of shape (B, T, dim) where channel k holds f0 * (k + 1) (fundamental and integer harmonics).
        if self.harmonic_num > 0:
            multipliers = mx.arange(1, self.dim + 1, dtype=f0_e.dtype)  # (dim,)
            f0_buf = f0_e * multipliers  # broadcast to (B, T, dim)
        else:
            f0_buf = f0_e  # (B, T, 1)

        rad_values = (f0_buf / self.sampling_rate) % 1  # (B, T, dim)

        # Sample (or accept) the per-batch, per-harmonic initial phase. The first column is forced to zero (matches
        # the reference) so that the fundamental component's phase is deterministic given f0.
        if rand_ini is None:
            rand_ini = mx.random.uniform(shape=(B, self.dim))
        # Zero out the first harmonic's initial phase. Build a (dim,) mask: [0, 1, 1, ..., 1].
        zero_first = mx.concatenate(
            [mx.zeros((1,), dtype=rand_ini.dtype), mx.ones((self.dim - 1,), dtype=rand_ini.dtype)],
            axis=0,
        ) if self.dim > 1 else mx.zeros((1,), dtype=rand_ini.dtype)
        rand_ini = rand_ini * zero_first
        # Add rand_ini to rad_values only at t=0.
        ini_at_t0 = mx.expand_dims(rand_ini, 1)  # (B, 1, dim)
        if T > 1:
            ini_pad = mx.concatenate(
                [ini_at_t0, mx.zeros((B, T - 1, self.dim), dtype=rad_values.dtype)],
                axis=1,
            )
        else:
            ini_pad = ini_at_t0
        rad_values = rad_values + ini_pad.astype(rad_values.dtype)

        # Cumulative phase, then upsample via linear (with align_corners=True) so the reconstructed sine is smooth.
        tmp_over_one = mx.cumsum(rad_values, axis=1) * upp
        tmp_over_one = interpolate_linear_axis(tmp_over_one, upp, axis=1)  # (B, T*upp, dim)

        # The "rad_values" themselves are repeated by nearest-neighbour, so each frame's phase increment is uniformly
        # spread across the `upp` audio samples that span the frame.
        rad_values_up = interpolate_nearest_axis(rad_values, upp, axis=1)  # (B, T*upp, dim)

        # Where the cumulative phase wraps past 1, inject a -1 shift so the running `cumsum` below stays continuous.
        tmp_over_one_mod = tmp_over_one % 1
        diff = tmp_over_one_mod[:, 1:, :] - tmp_over_one_mod[:, :-1, :]
        wrap_idx = (diff < 0).astype(rad_values_up.dtype)
        cumsum_shift = mx.concatenate(
            [mx.zeros((B, 1, self.dim), dtype=rad_values_up.dtype), -wrap_idx], axis=1
        )

        sine_waves = mx.sin(mx.cumsum(rad_values_up + cumsum_shift, axis=1) * (2 * math.pi))
        sine_waves = sine_waves * self.sine_amp

        # Voiced/unvoiced mask upsampled to audio rate via nearest-neighbour. Shape (B, T*upp, 1).
        uv = self._f02uv(f0_e)  # (B, T, 1)
        uv = interpolate_nearest_axis(uv, upp, axis=1)  # (B, T*upp, 1)

        if noise_raw is None:
            noise_raw = mx.random.normal(sine_waves.shape).astype(sine_waves.dtype)
        # noise_amp blends the explicit noise std (voiced regions) with a fraction of sine_amp (unvoiced "breath").
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * noise_raw

        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """
    Mixes the harmonic sine bank produced by `SineGen` down to a single excitation channel via a Linear + tanh.

    The PyTorch reference also accepts an `is_half` flag and conditionally casts to half precision for the merge.
    We accept the flag for API parity but do the actual half-precision cast in the downstream generator if needed,
    keeping this module dtype-agnostic for clarity.
    """

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
        is_half: bool = False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        self.l_sin_gen = SineGen(
            sampling_rate,
            harmonic_num=harmonic_num,
            sine_amp=sine_amp,
            noise_std=add_noise_std,
            voiced_threshold=voiced_threshold,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(
        self,
        x: mx.array,
        upp: int = 1,
        rand_ini: Optional[mx.array] = None,
        noise_raw: Optional[mx.array] = None,
    ) -> Tuple[mx.array, None, None]:
        sine_wavs, _, _ = self.l_sin_gen(x, upp, rand_ini=rand_ini, noise_raw=noise_raw)
        # The reference casts sine_wavs to the Linear's weight dtype; MLX broadcasts mixed-dtype matmul, so this is
        # implicit. We still match the reference's return shape `(sine_merge, None, None)` for API compatibility.
        sine_merge = mx.tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


# ----------------------------------------------------------------------------------------------------------------------
# NSF generator (ResBlock1, GeneratorNSF).
#
# `ResBlock1` is the HiFi-GAN-style residual block used inside the generator. The PyTorch reference wraps each Conv1d
# in `weight_norm` for training; the released RVC checkpoints have weight_norm fused back into plain weights via
# `remove_weight_norm()` before export, so the MLX side uses plain `nn.Conv1d`. The same applies to `GeneratorNSF.ups`
# (ConvTranspose1d).
# ----------------------------------------------------------------------------------------------------------------------


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Same-length padding for a 1D conv: `(kernel * dilation - dilation) // 2`. Matches RVC's `get_padding`."""
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    """
    HiFi-GAN-style residual block with three (Conv1d -> LeakyReLU -> Conv1d -> add) stages. The first conv in each
    stage uses one of the configured dilations; the second uses dilation=1. Padding is set so each conv preserves the
    time length.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = tuple(dilation)
        self.lrelu_slope = 0.1
        self.convs1 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=d,
                padding=_get_padding(kernel_size, d),
            )
            for d in self.dilation
        ]
        self.convs2 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=1,
                padding=_get_padding(kernel_size, 1),
            )
            for _ in self.dilation
        ]

    def __call__(self, x: mx.array, x_mask: Optional[mx.array] = None) -> mx.array:
        # x: (B, T, C); x_mask: (B, T, 1) or None
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = nn.leaky_relu(xt, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class GeneratorNSF(nn.Module):
    """
    Neural Source-Filter generator. Consumes a per-frame latent `x` plus a per-frame fundamental frequency `f0` and an
    optional speaker conditioning `g`, and produces audio samples at `T * prod(upsample_rates)` rate.

    Pipeline (channels-last MLX shapes):
      * `m_source(f0, upp)` -> harmonic excitation `(B, T * upp, 1)`
      * `conv_pre(x)` -> `(B, T, upsample_initial_channel)`
      * For each upsampling level i:
          - LeakyReLU + ConvTranspose1d to multiply time by `upsample_rates[i]`
          - Add `noise_convs[i](har_source)` (har_source pooled to match the current time length)
          - Sum-and-average over `num_kernels` parallel `ResBlock1`s
      * Final LeakyReLU + 7x1 Conv + tanh -> `(B, T_audio, 1)`

    Only `resblock="1"` (ResBlock1) is supported; the alternative ResBlock2 isn't used by the released RVC models.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
        upsample_rates: Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Tuple[int, ...],
        gin_channels: int,
        sr: int,
        is_half: bool = False,
    ):
        super().__init__()
        if resblock != "1":
            raise NotImplementedError("Only ResBlock1 ('1') is supported for the NSF generator.")
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsample_rates = tuple(upsample_rates)
        self.upp = math.prod(upsample_rates)
        self.gin_channels = gin_channels
        self.lrelu_slope = 0.1

        self.m_source = SourceModuleHnNSF(sampling_rate=sr, harmonic_num=0, is_half=is_half)
        # 7x1 input projection from initial_channel to upsample_initial_channel.
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3
        )

        self.ups: list = []
        self.noise_convs: list = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            if i + 1 < len(upsample_rates):
                # `stride_f0` is the time-compression factor needed so the noise channel matches the current x length.
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        out_ch,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                # At the last layer the noise source already matches the audio rate; a 1x1 conv is enough.
                self.noise_convs.append(nn.Conv1d(1, out_ch, kernel_size=1))

        # `resblocks` is a flat list of length `num_upsamples * num_kernels`; level i uses indices
        # [i * num_kernels, (i + 1) * num_kernels).
        self.resblocks: list = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, tuple(d)))

        final_ch = upsample_initial_channel // (2 ** len(upsample_rates))
        self.conv_post = nn.Conv1d(
            final_ch, 1, kernel_size=7, stride=1, padding=3, bias=False
        )

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, kernel_size=1)

    def __call__(
        self,
        x: mx.array,
        f0: mx.array,
        g: Optional[mx.array] = None,
        rand_ini: Optional[mx.array] = None,
        noise_raw: Optional[mx.array] = None,
    ) -> mx.array:
        # x: (B, T, initial_channel); f0: (B, T); g: (B, 1, gin_channels) or None
        har_source, _, _ = self.m_source(f0, self.upp, rand_ini=rand_ini, noise_raw=noise_raw)
        # har_source is already (B, T * upp, 1) in channels-last, which is what the noise_convs Conv1d consumes.

        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source

            # Sum-and-average over the parallel ResBlock1s at this level. The `xs is None` branch is the first one;
            # subsequent iterations accumulate.
            block_start = i * self.num_kernels
            xs = None
            for j in range(self.num_kernels):
                rb_out = self.resblocks[block_start + j](x)
                xs = rb_out if xs is None else xs + rb_out
            x = xs / self.num_kernels

        x = nn.leaky_relu(x)
        x = self.conv_post(x)
        x = mx.tanh(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
# Flow modules (WN, Flip, ResidualCouplingLayer, ResidualCouplingBlock).
#
# RVC uses a normalizing-flow stack between the text encoder posterior and the generator. At inference time the block
# runs in reverse mode, applying the inverse flow to draw `z ~ q(z|condition)`.
# ----------------------------------------------------------------------------------------------------------------------


def _fused_add_tanh_sigmoid_multiply(
    input_a: mx.array, input_b: mx.array, n_channels: int
) -> mx.array:
    """
    Gated activation: `tanh(left_half(a + b)) * sigmoid(right_half(a + b))`. The "left/right half" split is along the
    channel axis (which is the last axis in MLX channels-last).
    """
    in_act = input_a + input_b
    t_act = mx.tanh(in_act[..., :n_channels])
    s_act = mx.sigmoid(in_act[..., n_channels:])
    return t_act * s_act


class WN(nn.Module):
    """
    WaveNet-style dilated convolutional block used as the parameter-producing network inside `ResidualCouplingLayer`.

    Each layer:
      * runs a dilated 1D conv with double the hidden channels (so we have both the tanh and sigmoid halves);
      * optionally adds a slice of the speaker conditioning (one chunk per layer);
      * applies the fused tanh-sigmoid gate;
      * splits a 1x1 "res_skip" conv into a residual (added back into `x`) and a skip (accumulated into `output`).
        The last layer has no residual half.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "WN expects odd kernel_size."
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = float(p_dropout)

        self.drop = nn.Dropout(self.p_dropout)
        self.in_layers: list = []
        self.res_skip_layers: list = []

        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, kernel_size=1
            )

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(
                nn.Conv1d(hidden_channels, res_skip_channels, kernel_size=1)
            )

    def __call__(
        self, x: mx.array, x_mask: mx.array, g: Optional[mx.array] = None
    ) -> mx.array:
        # x: (B, T, hidden_channels); x_mask: (B, T, 1); g: (B, T_g, gin_channels) or None.
        output = mx.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)  # (B, T_g, 2 * hidden * n_layers)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)  # (B, T, 2 * hidden)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[..., cond_offset : cond_offset + 2 * self.hidden_channels]
            else:
                g_l = mx.zeros_like(x_in)

            acts = _fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[..., : self.hidden_channels]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[..., self.hidden_channels :]
            else:
                output = output + res_skip_acts
        return output * x_mask


class Flip(nn.Module):
    """
    Channel-axis flip used between coupling layers so consecutive layers transform the other half of the input.
    Carries no parameters. In `reverse=False` mode returns `(x_flipped, logdet=zeros(B,))`; in `reverse=True` mode
    returns `(x_flipped, zeros(1))` to match the RVC reference contract.
    """

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        g: Optional[mx.array] = None,
        reverse: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        # Channel axis is last in MLX channels-last.
        x = x[..., ::-1]
        if not reverse:
            logdet = mx.zeros((x.shape[0],), dtype=x.dtype)
            return x, logdet
        return x, mx.zeros((1,))


class ResidualCouplingLayer(nn.Module):
    """
    Affine coupling layer: splits `x` into two halves along the channel axis, transforms one half conditioned on the
    other via the parameter network `enc` (a `WN`), and concatenates them back. With `mean_only=True` the affine is a
    pure shift (logs == 0) — this is the variant used by `ResidualCouplingBlock` in RVC's synthesizer.

    Inference uses `reverse=True`: the inverse transformation `x1 = (x1 - m) * exp(-logs) * x_mask`.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()
        assert channels % 2 == 0, "channels must be divisible by 2."
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, kernel_size=1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=float(p_dropout),
            gin_channels=gin_channels,
        )
        # The reference zero-initializes both `post.weight` and `post.bias`. We follow suit so a freshly built module
        # is the identity transformation at the affine step (m == 0, logs == 0). Tests that compare against the torch
        # ref bypass this by copying weights across.
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - int(mean_only)), kernel_size=1
        )
        self.post.weight = mx.zeros_like(self.post.weight)
        if self.post.bias is not None:
            self.post.bias = mx.zeros_like(self.post.bias)

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        g: Optional[mx.array] = None,
        reverse: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        # x: (B, T, channels); split along channel axis.
        x0 = x[..., : self.half_channels]
        x1 = x[..., self.half_channels :]
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m = stats[..., : self.half_channels]
            logs = stats[..., self.half_channels :]
        else:
            m = stats
            logs = mx.zeros_like(m)

        if not reverse:
            x1 = m + x1 * mx.exp(logs) * x_mask
            x = mx.concatenate([x0, x1], axis=-1)
            # logdet summed over the time + channel axes (axes 1 and 2 in channels-last).
            logdet = mx.sum(logs, axis=(1, 2))
            return x, logdet
        x1 = (x1 - m) * mx.exp(-logs) * x_mask
        x = mx.concatenate([x0, x1], axis=-1)
        return x, mx.zeros((1,))


class ResidualCouplingBlock(nn.Module):
    """
    Stack of `n_flows` `ResidualCouplingLayer` modules (always `mean_only=True`) interleaved with `Flip`s. At inference
    time the block is applied in reverse: it consumes `z_p ~ N(m_p, exp(logs_p))` from the text encoder posterior and
    produces `z` that conditions the generator.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows: list = []
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        g: Optional[mx.array] = None,
        reverse: bool = False,
    ) -> mx.array:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in self.flows[::-1]:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x


# ----------------------------------------------------------------------------------------------------------------------
# Top-level synthesizer (inference path only).
#
# `SynthesizerTrnMs768NSFsid` chains the four learned components used at inference: text encoder, normalizing flow,
# generator, and speaker-embedding lookup. The training-time posterior encoder (`enc_q` in the reference) is omitted
# here — it's not used at inference and would need extra plumbing.
# ----------------------------------------------------------------------------------------------------------------------


# Standard noise temperature used by the RVC reference when sampling from the text-encoder posterior. Hard-coded to
# match the published implementation; not part of the learned config.
_INFER_NOISE_SCALE = 0.66666


class SynthesizerTrnMs768NSFsid(nn.Module):
    """
    The released RVC voice-conversion synthesizer (inference-only port).

    Pipeline at inference:
      1. `enc_p(phone, pitch, lengths)` -> posterior parameters `(m_p, logs_p)` and mask `x_mask`.
      2. Sample `z_p ~ N(m_p, exp(logs_p))` with temperature `_INFER_NOISE_SCALE`, mask out invalid positions.
      3. `flow(z_p, x_mask, g=g, reverse=True)` -> `z` in the prior space.
      4. `dec(z * x_mask, nsff0, g=g)` -> generated audio.

    For tests and deterministic conversion, the random tensors used at step 2 (`noise_z`) and inside the NSF source
    module (`rand_ini`, `noise_raw`) are exposed as optional kwargs to `.infer`. Production callers can leave them as
    `None` and get the original random behaviour.

    The `sr` argument accepts either an integer sampling rate or a string shorthand (`"32k"` / `"40k"` / `"48k"`),
    matching the reference's convenience handling.
    """

    _SR_ALIAS = {"32k": 32000, "40k": 40000, "48k": 48000}

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
        upsample_rates: Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Tuple[int, ...],
        spk_embed_dim: int,
        gin_channels: int,
        sr,
        **kwargs,
    ):
        super().__init__()
        if isinstance(sr, str):
            sr = self._SR_ALIAS[sr]
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = tuple(resblock_kernel_sizes)
        self.resblock_dilation_sizes = tuple(tuple(d) for d in resblock_dilation_sizes)
        self.upsample_rates = tuple(upsample_rates)
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = tuple(upsample_kernel_sizes)
        self.spk_embed_dim = spk_embed_dim
        self.gin_channels = gin_channels
        self.sr = sr

        self.enc_p = TextEncoder768(
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=float(p_dropout),
        )
        self.dec = GeneratorNSF(
            initial_channel=inter_channels,
            resblock=resblock,
            resblock_kernel_sizes=tuple(resblock_kernel_sizes),
            resblock_dilation_sizes=tuple(tuple(d) for d in resblock_dilation_sizes),
            upsample_rates=tuple(upsample_rates),
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=tuple(upsample_kernel_sizes),
            gin_channels=gin_channels,
            sr=sr,
            is_half=False,
        )
        self.flow = ResidualCouplingBlock(
            channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            gin_channels=gin_channels,
        )
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def infer(
        self,
        phone: mx.array,
        phone_lengths: mx.array,
        pitch: mx.array,
        nsff0: mx.array,
        sid: mx.array,
        max_len: Optional[int] = None,
        *,
        noise_z: Optional[mx.array] = None,
        rand_ini: Optional[mx.array] = None,
        noise_raw: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, Tuple[mx.array, mx.array, mx.array, mx.array]]:
        """
        Run the full inference pipeline.

        :param phone: per-frame content features, shape `(B, T, 768)`.
        :param phone_lengths: valid lengths along the time axis, shape `(B,)`.
        :param pitch: per-frame integer pitch class in `[0, 255]`, shape `(B, T)`.
        :param nsff0: per-frame continuous f0 in Hz, shape `(B, T)`.
        :param sid: per-batch speaker index, shape `(B,)`.
        :param max_len: optionally truncate the flow output to this many frames before feeding the generator.
        :param noise_z: standard-normal noise of shape `(B, T, inter_channels)` used for the posterior sample. Sampled
            internally when `None`.
        :param rand_ini: SineGen initial-phase noise of shape `(B, dim)`. Sampled internally when `None`.
        :param noise_raw: SineGen unvoiced-region noise of shape `(B, T * upp, dim)`. Sampled internally when `None`.
        :returns: a tuple `(o, x_mask, (z, z_p, m_p, logs_p))` where `o` is the generated audio of shape
            `(B, T_audio, 1)`.
        """
        # Speaker embedding broadcast across time. (B,) -> (B, gin_channels) -> (B, 1, gin_channels).
        g = mx.expand_dims(self.emb_g(sid), axis=1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        if noise_z is None:
            noise_z = mx.random.normal(m_p.shape).astype(m_p.dtype)
        z_p = (m_p + mx.exp(logs_p) * noise_z * _INFER_NOISE_SCALE) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        z_masked = z * x_mask
        if max_len is not None:
            z_masked = z_masked[:, :max_len, :]
            nsff0 = nsff0[:, :max_len]
        o = self.dec(z_masked, nsff0, g=g, rand_ini=rand_ini, noise_raw=noise_raw)
        return o, x_mask, (z, z_p, m_p, logs_p)
