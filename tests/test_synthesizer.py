"""
Paired-module tests for the RVC voice-conversion synthesizer building blocks.

Each test builds a PyTorch reference and an MLX implementation with matched init params and identical (copied)
weights, then compares forward outputs. The PyTorch reference uses channels-first conventions (`(B, C, T)`); the MLX
modules use channels-last (`(B, T, C)`), so the wrappers transpose at the boundary.

This file currently covers `LayerNorm`, `FFN`, `MultiHeadAttention`, and the transformer `Encoder`. Higher-level
synthesizer pieces (TextEncoder768, GeneratorNSF, ...) will arrive with their own tests in follow-up commits.
"""

import numpy as np
import pytest
import torch

from rvc_mlx.synthesizer import (
    Encoder,
    FFN,
    LayerNorm,
    MultiHeadAttention,
    TextEncoder768,
)
from rvc_mlx._torch_ref import (
    TorchFFN,
    TorchLayerNorm,
    TorchMultiHeadAttention,
    TorchTextEncoder768,
    TorchTransformerEncoder,
)

from .mlx_torch_comparison_framework import BaseOperationTest, OperationTestSuite
from .torch_bridge import (
    copy_ffn,
    copy_layer_norm,
    copy_multi_head_attention,
    copy_text_encoder_768,
    copy_transformer_encoder,
    set_eval,
    to_time_first,
    to_time_last,
)


# ----------------------------------------------------------------------------------------------------------------------
# LayerNorm. The MLX module normalizes over the last axis directly (channels-last), the PyTorch reference transposes
# channel to last, normalizes, and transposes back. Outputs in channels-first should match exactly.
# ----------------------------------------------------------------------------------------------------------------------


def _build_layer_norm_pair(channels, eps=1e-5, seed=0):
    torch.manual_seed(seed)
    t_ln = TorchLayerNorm(channels, eps=eps)
    # Randomize gamma/beta away from the identity init so the test exercises the affine path.
    with torch.no_grad():
        t_ln.gamma.copy_(torch.randn_like(t_ln.gamma) + 1.0)
        t_ln.beta.copy_(torch.randn_like(t_ln.beta))
    m_ln = LayerNorm(channels, eps=eps)
    copy_layer_norm(t_ln, m_ln)
    set_eval(t_ln, m_ln)

    def mlx_fn(x):
        return to_time_first(m_ln(to_time_last(x)))

    def torch_fn(x):
        with torch.no_grad():
            return t_ln(x)

    return mlx_fn, torch_fn


class TestSynthesizerLayerNorm(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_layer_norm_pair(channels=192)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "layer_norm")

        rng = np.random.default_rng(0)
        cls.suite.add_test_case(
            name="layer_norm_small",
            inputs={"x": rng.standard_normal((1, 192, 16)).astype(np.float32)},
            description="(B, C, T) = (1, 192, 16); RVC inter_channels-typical width",
            atol=1e-5,
            rtol=1e-5,
        )
        cls.suite.add_test_case(
            name="layer_norm_batch",
            inputs={"x": rng.standard_normal((4, 192, 32)).astype(np.float32)},
            description="Batched (B, C, T) = (4, 192, 32)",
            atol=1e-5,
            rtol=1e-5,
        )


# ----------------------------------------------------------------------------------------------------------------------
# FFN. Two Conv1d layers with an activation; relu or gelu-approx. Padded asymmetrically inside `padding` so a Conv1d
# of any kernel_size preserves the time length. The mask multiplies the input before each conv.
# ----------------------------------------------------------------------------------------------------------------------


def _build_ffn_pair(
    in_channels, out_channels, filter_channels, kernel_size, activation=None, causal=False, seed=0
):
    torch.manual_seed(seed)
    t_ffn = TorchFFN(
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=activation,
        causal=causal,
    )
    m_ffn = FFN(
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=activation,
        causal=causal,
    )
    copy_ffn(t_ffn, m_ffn)
    set_eval(t_ffn, m_ffn)

    def mlx_fn(x, x_mask):
        # x: (B, C, T) -> (B, T, C); x_mask: (B, 1, T) -> (B, T, 1)
        out = m_ffn(to_time_last(x), to_time_last(x_mask))
        return to_time_first(out)

    def torch_fn(x, x_mask):
        with torch.no_grad():
            return t_ffn(x, x_mask)

    return mlx_fn, torch_fn


def _mask_inputs(batch, channels, length, valid_lengths, rng):
    """Build an (x, x_mask) pair where x_mask has 1s up to valid_lengths[i] and 0s after."""
    x = rng.standard_normal((batch, channels, length)).astype(np.float32)
    mask = np.zeros((batch, 1, length), dtype=np.float32)
    for i, vl in enumerate(valid_lengths):
        mask[i, 0, :vl] = 1.0
    return x, mask


class TestSynthesizerFFNKernel1Relu(BaseOperationTest):
    """FFN with kernel_size=1 (no padding) and ReLU activation. Equivalent to two per-position Linear layers."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_ffn_pair(
            in_channels=192, out_channels=192, filter_channels=768, kernel_size=1
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "ffn_k1_relu")

        rng = np.random.default_rng(0)
        x, mask = _mask_inputs(2, 192, 16, [16, 10], rng)
        cls.suite.add_test_case(
            name="ffn_k1_relu_partial_mask",
            inputs={"x": x, "x_mask": mask},
            description="kernel_size=1, ReLU; second batch entry has 6 masked-out trailing frames",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerFFNKernel3Relu(BaseOperationTest):
    """FFN with kernel_size=3 (symmetric pad 1/1) and ReLU."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_ffn_pair(
            in_channels=192, out_channels=192, filter_channels=768, kernel_size=3
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "ffn_k3_relu")

        rng = np.random.default_rng(1)
        x, mask = _mask_inputs(1, 192, 32, [32], rng)
        cls.suite.add_test_case(
            name="ffn_k3_relu_full_mask",
            inputs={"x": x, "x_mask": mask},
            description="kernel_size=3 (RVC default), ReLU, full mask",
            atol=1e-4,
            rtol=1e-4,
        )

        x, mask = _mask_inputs(2, 192, 24, [24, 18], rng)
        cls.suite.add_test_case(
            name="ffn_k3_relu_partial_mask",
            inputs={"x": x, "x_mask": mask},
            description="kernel_size=3, partial mask; second batch entry has 6 trailing masked frames",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerFFNKernel3Gelu(BaseOperationTest):
    """FFN with the sigmoid-linear "GELU approximation" used when activation='gelu'."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_ffn_pair(
            in_channels=192,
            out_channels=192,
            filter_channels=768,
            kernel_size=3,
            activation="gelu",
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "ffn_k3_gelu")

        rng = np.random.default_rng(2)
        x, mask = _mask_inputs(1, 192, 16, [16], rng)
        cls.suite.add_test_case(
            name="ffn_k3_gelu",
            inputs={"x": x, "x_mask": mask},
            description="kernel_size=3 with x * sigmoid(1.702 * x) activation",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerFFNKernel4Causal(BaseOperationTest):
    """FFN with an even kernel (4) and causal padding (left-only). Catches asymmetric-padding bugs."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_ffn_pair(
            in_channels=64,
            out_channels=64,
            filter_channels=256,
            kernel_size=4,
            causal=True,
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "ffn_k4_causal")

        rng = np.random.default_rng(3)
        x, mask = _mask_inputs(1, 64, 12, [12], rng)
        cls.suite.add_test_case(
            name="ffn_k4_causal",
            inputs={"x": x, "x_mask": mask},
            description="kernel_size=4 (even) with causal=True; pads left by 3",
            atol=1e-4,
            rtol=1e-4,
        )


# ----------------------------------------------------------------------------------------------------------------------
# MultiHeadAttention. Self-attention path + relative-position embedding path, both with and without an attention mask.
# ----------------------------------------------------------------------------------------------------------------------


def _build_mha_pair(channels, n_heads, window_size=None, seed=0):
    torch.manual_seed(seed)
    t_mha = TorchMultiHeadAttention(
        channels=channels,
        out_channels=channels,
        n_heads=n_heads,
        p_dropout=0.0,
        window_size=window_size,
    )
    m_mha = MultiHeadAttention(
        channels=channels,
        out_channels=channels,
        n_heads=n_heads,
        p_dropout=0.0,
        window_size=window_size,
    )
    copy_multi_head_attention(t_mha, m_mha)
    set_eval(t_mha, m_mha)

    def mlx_fn(x, attn_mask=None):
        # x: (B, C, T) -> (B, T, C); attn_mask is left as-is, already in (B, 1, T, T) layout.
        out = m_mha(to_time_last(x), to_time_last(x), attn_mask=attn_mask)
        return to_time_first(out)

    def torch_fn(x, attn_mask=None):
        with torch.no_grad():
            return t_mha(x, x, attn_mask=attn_mask)

    return mlx_fn, torch_fn


def _self_attn_mask(batch, length, valid_lengths):
    """
    Build an attention mask of shape (B, 1, T, T) where entry (i, *, q, k) is 1 iff both q < valid[i] and k < valid[i].
    Matches the construction in `Encoder` (`x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)` in the PyTorch reference).
    """
    mask = np.zeros((batch, 1, length, length), dtype=np.float32)
    for i, vl in enumerate(valid_lengths):
        mask[i, 0, :vl, :vl] = 1.0
    return mask


class TestSynthesizerMultiHeadAttentionNoRel(BaseOperationTest):
    """MultiHeadAttention without relative-position embeddings."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_mha_pair(channels=192, n_heads=2, window_size=None)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "mha_no_rel")

        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 192, 16)).astype(np.float32)
        cls.suite.add_test_case(
            name="mha_no_rel_no_mask",
            inputs={"x": x},
            description="Self-attention, no mask, no relative bias",
            atol=1e-4,
            rtol=1e-4,
        )

        attn_mask = _self_attn_mask(1, 16, [12])
        cls.suite.add_test_case(
            name="mha_no_rel_partial_mask",
            inputs={"x": x, "attn_mask": attn_mask},
            description="Self-attention with the last 4 frames masked out",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerMultiHeadAttentionRel(BaseOperationTest):
    """MultiHeadAttention with window_size=10 relative-position embeddings (RVC default)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_mha_pair(channels=192, n_heads=2, window_size=10)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "mha_rel")

        rng = np.random.default_rng(1)
        # Use a sequence longer than window_size+1 to exercise the "no padding, only slice" branch.
        x_long = rng.standard_normal((1, 192, 24)).astype(np.float32)
        cls.suite.add_test_case(
            name="mha_rel_long_seq",
            inputs={"x": x_long, "attn_mask": _self_attn_mask(1, 24, [24])},
            description="Long sequence: relative embeddings get sliced, no padding",
            atol=1e-4,
            rtol=1e-4,
        )

        # Use a sequence shorter than window_size+1 to exercise the symmetric-padding branch.
        x_short = rng.standard_normal((1, 192, 6)).astype(np.float32)
        cls.suite.add_test_case(
            name="mha_rel_short_seq",
            inputs={"x": x_short, "attn_mask": _self_attn_mask(1, 6, [6])},
            description="Short sequence: relative embeddings get symmetrically padded before slicing",
            atol=1e-4,
            rtol=1e-4,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Encoder (transformer). Stacks attention + LayerNorm + FFN + LayerNorm with post-norm residual connections.
# ----------------------------------------------------------------------------------------------------------------------


def _build_encoder_pair(
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    window_size=10,
    seed=0,
):
    torch.manual_seed(seed)
    t_enc = TorchTransformerEncoder(
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=0.0,
        window_size=window_size,
    )
    m_enc = Encoder(
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=0.0,
        window_size=window_size,
    )
    copy_transformer_encoder(t_enc, m_enc)
    set_eval(t_enc, m_enc)

    def mlx_fn(x, x_mask):
        # x: (B, C, T) -> (B, T, C); x_mask: (B, 1, T) -> (B, T, 1)
        out = m_enc(to_time_last(x), to_time_last(x_mask))
        return to_time_first(out)

    def torch_fn(x, x_mask):
        with torch.no_grad():
            return t_enc(x, x_mask)

    return mlx_fn, torch_fn


class TestSynthesizerEncoderRvcDefaults(BaseOperationTest):
    """Full transformer encoder with RVC's default config (192/768/2 heads/6 layers, kernel=3, window=10)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_encoder_pair()
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "transformer_encoder")

        rng = np.random.default_rng(0)
        x, mask = _mask_inputs(1, 192, 24, [24], rng)
        cls.suite.add_test_case(
            name="encoder_full_mask",
            inputs={"x": x, "x_mask": mask},
            description="RVC-default encoder, fully valid 24-frame input",
            atol=1e-3,
            rtol=1e-3,
        )

        x, mask = _mask_inputs(2, 192, 32, [32, 20], rng)
        cls.suite.add_test_case(
            name="encoder_partial_mask",
            inputs={"x": x, "x_mask": mask},
            description="RVC-default encoder; second batch entry has 12 trailing masked frames",
            atol=1e-3,
            rtol=1e-3,
        )


# ----------------------------------------------------------------------------------------------------------------------
# TextEncoder768. Composes phone Linear + pitch Embedding + transformer Encoder + 1x1 projection.
# Returns a (m, logs, x_mask) tuple; the framework compares one array at a time, so we wrap each output in its own
# test class.
# ----------------------------------------------------------------------------------------------------------------------


# RVC's default text-encoder config (from `SynthesizerTrnMs768NSFsid.__init__`).
TEXT_ENCODER_768_DEFAULTS = dict(
    out_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0.0,
)


def _build_text_encoder_768_pair(seed=0, **init_overrides):
    init_params = {**TEXT_ENCODER_768_DEFAULTS, **init_overrides}
    torch.manual_seed(seed)
    t_te = TorchTextEncoder768(**init_params)
    m_te = TextEncoder768(**init_params)
    copy_text_encoder_768(t_te, m_te)
    set_eval(t_te, m_te)
    return t_te, m_te


def _make_text_encoder_768_wrappers(t_te, m_te, output_index: int):
    """
    The module returns `(m, logs, x_mask)`. `output_index` picks which tensor to expose (0/1/2). The PyTorch result is
    channels-first; the MLX result is channels-last. We transpose the MLX side to channels-first to compare.
    """

    def mlx_fn(phone, pitch, lengths):
        out = m_te(phone, pitch, lengths)[output_index]
        return to_time_first(out)

    def torch_fn(phone, pitch, lengths):
        with torch.no_grad():
            return t_te(phone, pitch, lengths)[output_index]

    return mlx_fn, torch_fn


def _text_encoder_inputs(batch, length, valid_lengths, rng):
    """Build (phone, pitch, lengths) for the text encoder tests."""
    phone = rng.standard_normal((batch, length, 768)).astype(np.float32)
    pitch = rng.integers(0, 256, size=(batch, length)).astype(np.int64)
    lengths = np.array(valid_lengths, dtype=np.int64)
    return phone, pitch, lengths


class TestSynthesizerTextEncoder768Mean(BaseOperationTest):
    """TextEncoder768 returning the `m` (mean) head of the posterior. Full-batch with mixed lengths."""

    @classmethod
    def setup_class(cls):
        t_te, m_te = _build_text_encoder_768_pair()
        mlx_fn, torch_fn = _make_text_encoder_768_wrappers(t_te, m_te, output_index=0)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "text_encoder_768_m")

        rng = np.random.default_rng(0)
        phone, pitch, lengths = _text_encoder_inputs(2, 32, [32, 20], rng)
        cls.suite.add_test_case(
            name="m_partial_mask",
            inputs={"phone": phone, "pitch": pitch, "lengths": lengths},
            description="Mean head; second batch entry has 12 trailing masked frames",
            atol=1e-3,
            rtol=1e-3,
        )


class TestSynthesizerTextEncoder768Logs(BaseOperationTest):
    """TextEncoder768 returning the `logs` (log-std) head of the posterior."""

    @classmethod
    def setup_class(cls):
        t_te, m_te = _build_text_encoder_768_pair()
        mlx_fn, torch_fn = _make_text_encoder_768_wrappers(t_te, m_te, output_index=1)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "text_encoder_768_logs")

        rng = np.random.default_rng(1)
        phone, pitch, lengths = _text_encoder_inputs(1, 24, [24], rng)
        cls.suite.add_test_case(
            name="logs_full_mask",
            inputs={"phone": phone, "pitch": pitch, "lengths": lengths},
            description="Log-std head; fully valid single-sequence batch",
            atol=1e-3,
            rtol=1e-3,
        )


class TestSynthesizerTextEncoder768Mask(BaseOperationTest):
    """TextEncoder768 returning the `x_mask` derived from `lengths`. Pure boundary check."""

    @classmethod
    def setup_class(cls):
        t_te, m_te = _build_text_encoder_768_pair()
        mlx_fn, torch_fn = _make_text_encoder_768_wrappers(t_te, m_te, output_index=2)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "text_encoder_768_xmask")

        rng = np.random.default_rng(2)
        phone, pitch, lengths = _text_encoder_inputs(3, 16, [16, 10, 7], rng)
        cls.suite.add_test_case(
            name="xmask_varied",
            inputs={"phone": phone, "pitch": pitch, "lengths": lengths},
            description="x_mask derived from per-batch lengths; verifies sequence_mask shape and dtype",
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
