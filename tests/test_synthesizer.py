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
    Flip,
    GeneratorNSF,
    LayerNorm,
    MultiHeadAttention,
    ResBlock1,
    ResidualCouplingBlock,
    ResidualCouplingLayer,
    SineGen,
    SourceModuleHnNSF,
    TextEncoder768,
    WN,
)
from rvc_mlx._torch_ref import (
    TorchFFN,
    TorchFlip,
    TorchGeneratorNSF,
    TorchLayerNorm,
    TorchMultiHeadAttention,
    TorchResBlock1,
    TorchResidualCouplingBlock,
    TorchResidualCouplingLayer,
    TorchSineGen,
    TorchSourceModuleHnNSF,
    TorchTextEncoder768,
    TorchTransformerEncoder,
    TorchWN,
)

from .mlx_torch_comparison_framework import BaseOperationTest, OperationTestSuite
from .torch_bridge import (
    copy_ffn,
    copy_generator_nsf,
    copy_layer_norm,
    copy_multi_head_attention,
    copy_res_block1,
    copy_residual_coupling_block,
    copy_residual_coupling_layer,
    copy_source_module_hn_nsf,
    copy_text_encoder_768,
    copy_transformer_encoder,
    copy_wn,
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


# ----------------------------------------------------------------------------------------------------------------------
# NSF source modules (SineGen, SourceModuleHnNSF).
#
# Both modules sample random tensors internally in the original reference. Our refactored versions accept `rand_ini`
# (per-harmonic initial-phase noise) and `noise_raw` (unit-variance Gaussian noise) as optional kwargs so paired-module
# tests can feed the same tensors to both impls.
# ----------------------------------------------------------------------------------------------------------------------


def _sine_gen_inputs(batch, length, upp, dim, rng):
    """Generate (f0, rand_ini, noise_raw) test inputs."""
    # Mix voiced (f0 > 0) and unvoiced (f0 == 0) regions to exercise both branches of _f02uv.
    f0 = rng.uniform(50.0, 500.0, size=(batch, length)).astype(np.float32)
    # Force the last few frames per batch to be unvoiced.
    f0[:, -4:] = 0.0
    rand_ini = rng.uniform(size=(batch, dim)).astype(np.float32)
    noise_raw = rng.standard_normal((batch, length * upp, dim)).astype(np.float32)
    return f0, rand_ini, noise_raw


def _build_sine_gen_pair(harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0.0):
    init = dict(
        samp_rate=16000,
        harmonic_num=harmonic_num,
        sine_amp=sine_amp,
        noise_std=noise_std,
        voiced_threshold=voiced_threshold,
    )
    t_sg = TorchSineGen(**init)
    m_sg = SineGen(**init)
    set_eval(t_sg, m_sg)
    return t_sg, m_sg


class TestSynthesizerSineGenFundamentalOnly(BaseOperationTest):
    """SineGen with harmonic_num=0 (RVC default): just the fundamental sine plus noise."""

    @classmethod
    def setup_class(cls):
        t_sg, m_sg = _build_sine_gen_pair(harmonic_num=0)

        def mlx_fn(f0, upp, rand_ini, noise_raw):
            sine, _, _ = m_sg(f0, upp, rand_ini=rand_ini, noise_raw=noise_raw)
            return sine

        def torch_fn(f0, upp, rand_ini, noise_raw):
            with torch.no_grad():
                sine, _, _ = t_sg(f0, upp, rand_ini=rand_ini, noise_raw=noise_raw)
            return sine

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "sine_gen_fund")

        rng = np.random.default_rng(0)
        f0, rand_ini, noise_raw = _sine_gen_inputs(batch=1, length=16, upp=4, dim=1, rng=rng)
        cls.suite.add_test_case(
            name="sine_gen_fund_upp4",
            inputs={"f0": f0, "upp": 4, "rand_ini": rand_ini, "noise_raw": noise_raw},
            description="Fundamental-only SineGen, upp=4, mixed voiced/unvoiced frames",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerSineGenHarmonics(BaseOperationTest):
    """SineGen with harmonic_num=2 to exercise the multi-channel harmonic-multiplier path."""

    @classmethod
    def setup_class(cls):
        t_sg, m_sg = _build_sine_gen_pair(harmonic_num=2)

        def mlx_fn(f0, upp, rand_ini, noise_raw):
            sine, _, _ = m_sg(f0, upp, rand_ini=rand_ini, noise_raw=noise_raw)
            return sine

        def torch_fn(f0, upp, rand_ini, noise_raw):
            with torch.no_grad():
                sine, _, _ = t_sg(f0, upp, rand_ini=rand_ini, noise_raw=noise_raw)
            return sine

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "sine_gen_harm2")

        rng = np.random.default_rng(1)
        f0, rand_ini, noise_raw = _sine_gen_inputs(batch=2, length=12, upp=4, dim=3, rng=rng)
        cls.suite.add_test_case(
            name="sine_gen_harm2_upp4",
            inputs={"f0": f0, "upp": 4, "rand_ini": rand_ini, "noise_raw": noise_raw},
            description="SineGen with 2 harmonics (dim=3 total channels), upp=4",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerSineGenUV(BaseOperationTest):
    """Voiced/unvoiced output of SineGen. Doesn't depend on noise; should match exactly."""

    @classmethod
    def setup_class(cls):
        t_sg, m_sg = _build_sine_gen_pair(harmonic_num=0, voiced_threshold=10.0)

        def mlx_fn(f0, upp, rand_ini, noise_raw):
            _, uv, _ = m_sg(f0, upp, rand_ini=rand_ini, noise_raw=noise_raw)
            return uv

        def torch_fn(f0, upp, rand_ini, noise_raw):
            with torch.no_grad():
                _, uv, _ = t_sg(f0, upp, rand_ini=rand_ini, noise_raw=noise_raw)
            return uv

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "sine_gen_uv")

        rng = np.random.default_rng(2)
        f0, rand_ini, noise_raw = _sine_gen_inputs(batch=1, length=16, upp=4, dim=1, rng=rng)
        # Tighten f0 so some frames are below the 10 Hz threshold and others above.
        f0[0, ::3] = 5.0  # every third frame is below threshold -> unvoiced
        cls.suite.add_test_case(
            name="sine_gen_uv_threshold",
            inputs={"f0": f0, "upp": 4, "rand_ini": rand_ini, "noise_raw": noise_raw},
            description="voiced_threshold=10 with mixed below/above-threshold frames",
            atol=1e-6,
            rtol=1e-6,
        )


# ----------------------------------------------------------------------------------------------------------------------
# SourceModuleHnNSF wraps SineGen with a Linear + tanh that mixes harmonics down to a single channel.
# ----------------------------------------------------------------------------------------------------------------------


def _build_source_module_pair(harmonic_num=0, seed=0):
    torch.manual_seed(seed)
    init = dict(
        sampling_rate=16000,
        harmonic_num=harmonic_num,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshold=0.0,
        is_half=False,
    )
    t_sm = TorchSourceModuleHnNSF(**init)
    m_sm = SourceModuleHnNSF(**init)
    copy_source_module_hn_nsf(t_sm, m_sm)
    set_eval(t_sm, m_sm)

    def mlx_fn(x, upp, rand_ini, noise_raw):
        sine_merge, _, _ = m_sm(x, upp, rand_ini=rand_ini, noise_raw=noise_raw)
        return sine_merge

    def torch_fn(x, upp, rand_ini, noise_raw):
        with torch.no_grad():
            sine_merge, _, _ = t_sm(x, upp, rand_ini=rand_ini, noise_raw=noise_raw)
        return sine_merge

    return mlx_fn, torch_fn


class TestSynthesizerSourceModuleFund(BaseOperationTest):
    """SourceModuleHnNSF with harmonic_num=0 (RVC default for the inference NSF generator)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_source_module_pair(harmonic_num=0)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "source_module_fund")

        rng = np.random.default_rng(0)
        f0, rand_ini, noise_raw = _sine_gen_inputs(batch=1, length=16, upp=4, dim=1, rng=rng)
        cls.suite.add_test_case(
            name="source_module_fund_upp4",
            inputs={"x": f0, "upp": 4, "rand_ini": rand_ini, "noise_raw": noise_raw},
            description="Fundamental-only source module, upp=4, mixed voicing",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerSourceModuleHarmonics(BaseOperationTest):
    """SourceModuleHnNSF with harmonic_num=2 to exercise the multi-harmonic Linear projection."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_source_module_pair(harmonic_num=2, seed=1)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "source_module_harm2")

        rng = np.random.default_rng(1)
        f0, rand_ini, noise_raw = _sine_gen_inputs(batch=2, length=12, upp=4, dim=3, rng=rng)
        cls.suite.add_test_case(
            name="source_module_harm2_upp4",
            inputs={"x": f0, "upp": 4, "rand_ini": rand_ini, "noise_raw": noise_raw},
            description="Source module with 2 harmonics (dim=3 -> Linear projects to 1 channel)",
            atol=1e-4,
            rtol=1e-4,
        )


# ----------------------------------------------------------------------------------------------------------------------
# ResBlock1 and GeneratorNSF.
#
# Both wrap several Conv1d / ConvTranspose1d layers and have no internal randomness, so paired-module testing is
# straightforward once weights are copied.
# ----------------------------------------------------------------------------------------------------------------------


def _build_res_block1_pair(channels=32, kernel_size=3, dilation=(1, 3, 5), seed=0):
    torch.manual_seed(seed)
    t_rb = TorchResBlock1(channels, kernel_size, dilation)
    m_rb = ResBlock1(channels, kernel_size, dilation)
    copy_res_block1(t_rb, m_rb)
    set_eval(t_rb, m_rb)

    def mlx_fn(x, x_mask=None):
        out = m_rb(to_time_last(x), to_time_last(x_mask) if x_mask is not None else None)
        return to_time_first(out)

    def torch_fn(x, x_mask=None):
        with torch.no_grad():
            return t_rb(x, x_mask)

    return mlx_fn, torch_fn


class TestSynthesizerResBlock1Default(BaseOperationTest):
    """ResBlock1 with the RVC default (kernel=3, dilations=(1, 3, 5))."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_res_block1_pair(channels=32)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "resblock1_default")

        rng = np.random.default_rng(0)
        cls.suite.add_test_case(
            name="resblock1_no_mask",
            inputs={"x": rng.standard_normal((1, 32, 24)).astype(np.float32)},
            description="Default ResBlock1, no mask",
            atol=1e-4,
            rtol=1e-4,
        )

        x = rng.standard_normal((2, 32, 24)).astype(np.float32)
        mask = np.zeros((2, 1, 24), dtype=np.float32)
        mask[0, 0, :24] = 1.0
        mask[1, 0, :18] = 1.0
        cls.suite.add_test_case(
            name="resblock1_partial_mask",
            inputs={"x": x, "x_mask": mask},
            description="ResBlock1 with partial mask zeroing the trailing frames of the second batch entry",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerResBlock1K7(BaseOperationTest):
    """ResBlock1 with kernel=7, exercising larger dilated receptive fields."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_res_block1_pair(channels=16, kernel_size=7, dilation=(1, 3, 5), seed=1)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "resblock1_k7")

        rng = np.random.default_rng(1)
        cls.suite.add_test_case(
            name="resblock1_k7_no_mask",
            inputs={"x": rng.standard_normal((1, 16, 32)).astype(np.float32)},
            description="ResBlock1 with kernel=7 dilations (1, 3, 5)",
            atol=1e-4,
            rtol=1e-4,
        )


# Minimal NSF generator config for tests. The real RVC defaults (initial_channel=192, upsample_initial_channel=512,
# upsample_rates=[10, 10, 2, 2]) are too large for quick CI tests; this scaled-down config still exercises every code
# path (multi-level upsampling, multi-kernel ResBlock, optional speaker conditioning).
_GEN_TEST_CONFIG = dict(
    initial_channel=8,
    resblock="1",
    resblock_kernel_sizes=(3, 7),
    resblock_dilation_sizes=((1, 3, 5), (1, 3, 5)),
    upsample_rates=(4, 2),
    upsample_initial_channel=32,
    upsample_kernel_sizes=(8, 4),
    gin_channels=4,
    sr=16000,
    is_half=False,
)


def _build_generator_nsf_pair(seed=0, **overrides):
    config = {**_GEN_TEST_CONFIG, **overrides}
    torch.manual_seed(seed)
    t_gen = TorchGeneratorNSF(**config)
    m_gen = GeneratorNSF(**config)
    copy_generator_nsf(t_gen, m_gen)
    set_eval(t_gen, m_gen)

    upp = int(np.prod(config["upsample_rates"]))
    return t_gen, m_gen, config, upp


def _generator_inputs(batch, T, initial_channel, gin_channels, upp, rng):
    """Build (x, f0, g, rand_ini, noise_raw) for the generator test."""
    x = rng.standard_normal((batch, initial_channel, T)).astype(np.float32)
    f0 = rng.uniform(50.0, 500.0, size=(batch, T)).astype(np.float32)
    f0[:, -2:] = 0.0  # last frames unvoiced
    g = rng.standard_normal((batch, gin_channels, 1)).astype(np.float32)
    rand_ini = rng.uniform(size=(batch, 1)).astype(np.float32)  # dim == harmonic_num + 1 = 1
    noise_raw = rng.standard_normal((batch, T * upp, 1)).astype(np.float32)
    return x, f0, g, rand_ini, noise_raw


class TestSynthesizerGeneratorNSFWithSpeaker(BaseOperationTest):
    """GeneratorNSF with non-zero gin_channels (i.e., with speaker conditioning)."""

    @classmethod
    def setup_class(cls):
        t_gen, m_gen, config, upp = _build_generator_nsf_pair()

        def mlx_fn(x, f0, g, rand_ini, noise_raw):
            # x: (B, C, T) -> (B, T, C); g: (B, gin, 1) -> (B, 1, gin)
            out = m_gen(
                to_time_last(x),
                f0,
                g=to_time_last(g),
                rand_ini=rand_ini,
                noise_raw=noise_raw,
            )
            # out: (B, T_audio, 1) -> (B, 1, T_audio)
            return to_time_first(out)

        def torch_fn(x, f0, g, rand_ini, noise_raw):
            with torch.no_grad():
                return t_gen(x, f0, g=g, rand_ini=rand_ini, noise_raw=noise_raw)

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "generator_nsf_with_g")

        rng = np.random.default_rng(0)
        x, f0, g, rand_ini, noise_raw = _generator_inputs(
            batch=1, T=8, initial_channel=config["initial_channel"], gin_channels=config["gin_channels"], upp=upp, rng=rng
        )
        cls.suite.add_test_case(
            name="generator_with_speaker",
            inputs={"x": x, "f0": f0, "g": g, "rand_ini": rand_ini, "noise_raw": noise_raw},
            description="GeneratorNSF with 2-level upsampling (4x, 2x), 2 parallel ResBlocks per level, with speaker g",
            atol=1e-3,
            rtol=1e-3,
        )


class TestSynthesizerGeneratorNSFNoSpeaker(BaseOperationTest):
    """GeneratorNSF with gin_channels=0 (no speaker conditioning; cond layer is absent)."""

    @classmethod
    def setup_class(cls):
        t_gen, m_gen, config, upp = _build_generator_nsf_pair(seed=1, gin_channels=0)

        def mlx_fn(x, f0, rand_ini, noise_raw):
            out = m_gen(to_time_last(x), f0, rand_ini=rand_ini, noise_raw=noise_raw)
            return to_time_first(out)

        def torch_fn(x, f0, rand_ini, noise_raw):
            with torch.no_grad():
                return t_gen(x, f0, rand_ini=rand_ini, noise_raw=noise_raw)

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "generator_nsf_no_g")

        rng = np.random.default_rng(2)
        # gin_channels=0 so no g.
        x, f0, _g, rand_ini, noise_raw = _generator_inputs(
            batch=1, T=8, initial_channel=config["initial_channel"], gin_channels=1, upp=upp, rng=rng
        )
        cls.suite.add_test_case(
            name="generator_no_speaker",
            inputs={"x": x, "f0": f0, "rand_ini": rand_ini, "noise_raw": noise_raw},
            description="GeneratorNSF with gin_channels=0 (no speaker conditioning path)",
            atol=1e-3,
            rtol=1e-3,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Flow modules (WN, Flip, ResidualCouplingLayer, ResidualCouplingBlock).
# ----------------------------------------------------------------------------------------------------------------------


def _build_wn_pair(
    hidden_channels=64, kernel_size=5, dilation_rate=1, n_layers=3, gin_channels=0, seed=0
):
    torch.manual_seed(seed)
    init = dict(
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        p_dropout=0.0,
    )
    t_wn = TorchWN(**init)
    m_wn = WN(**init)
    copy_wn(t_wn, m_wn)
    set_eval(t_wn, m_wn)

    def mlx_fn(x, x_mask, g=None):
        out = m_wn(
            to_time_last(x),
            to_time_last(x_mask),
            g=to_time_last(g) if g is not None else None,
        )
        return to_time_first(out)

    def torch_fn(x, x_mask, g=None):
        with torch.no_grad():
            return t_wn(x, x_mask, g=g)

    return mlx_fn, torch_fn


class TestSynthesizerWNNoCond(BaseOperationTest):
    """WN without speaker conditioning (cond_layer absent)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_wn_pair(gin_channels=0)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "wn_no_cond")

        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 64, 24)).astype(np.float32)
        mask = np.ones((1, 1, 24), dtype=np.float32)
        cls.suite.add_test_case(
            name="wn_no_cond_basic",
            inputs={"x": x, "x_mask": mask},
            description="WN forward without speaker conditioning, full mask",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerWNWithCond(BaseOperationTest):
    """WN with speaker conditioning (`gin_channels != 0`)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_wn_pair(gin_channels=8, seed=1)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "wn_with_cond")

        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 64, 24)).astype(np.float32)
        mask = np.zeros((1, 1, 24), dtype=np.float32)
        mask[0, 0, :20] = 1.0
        # Speaker embedding broadcast across time: (B, gin, 1).
        g = rng.standard_normal((1, 8, 1)).astype(np.float32)
        cls.suite.add_test_case(
            name="wn_with_cond_partial_mask",
            inputs={"x": x, "x_mask": mask, "g": g},
            description="WN with speaker conditioning broadcast across time, partial mask",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerFlip(BaseOperationTest):
    """Flip module: reverses channels with no parameters. Tests both forward and reverse modes."""

    @classmethod
    def setup_class(cls):
        t_fl = TorchFlip()
        m_fl = Flip()
        set_eval(t_fl, m_fl)

        def mlx_fn(x, x_mask, reverse=False):
            out, _ = m_fl(to_time_last(x), to_time_last(x_mask), reverse=reverse)
            return to_time_first(out)

        def torch_fn(x, x_mask, reverse=False):
            with torch.no_grad():
                out, _ = t_fl(x, x_mask, reverse=reverse)
            return out

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "flip")

        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 8, 12)).astype(np.float32)
        mask = np.ones((1, 1, 12), dtype=np.float32)
        cls.suite.add_test_case(
            name="flip_forward",
            inputs={"x": x, "x_mask": mask, "reverse": False},
            description="Flip in forward mode reverses channels",
            atol=0.0,
            rtol=0.0,
        )
        cls.suite.add_test_case(
            name="flip_reverse",
            inputs={"x": x, "x_mask": mask, "reverse": True},
            description="Flip in reverse mode reverses channels (same result as forward)",
            atol=0.0,
            rtol=0.0,
        )


def _build_residual_coupling_layer_pair(
    channels=8, hidden_channels=16, kernel_size=5, dilation_rate=1, n_layers=2,
    gin_channels=0, mean_only=False, seed=0,
):
    torch.manual_seed(seed)
    init = dict(
        channels=channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        gin_channels=gin_channels,
        mean_only=mean_only,
    )
    t_rcl = TorchResidualCouplingLayer(**init)
    # Randomize post weights/biases (the reference zero-inits these, which would make the test compare zeros).
    with torch.no_grad():
        t_rcl.post.weight.copy_(torch.randn_like(t_rcl.post.weight) * 0.1)
        t_rcl.post.bias.copy_(torch.randn_like(t_rcl.post.bias) * 0.1)
    m_rcl = ResidualCouplingLayer(**init)
    copy_residual_coupling_layer(t_rcl, m_rcl)
    set_eval(t_rcl, m_rcl)

    def mlx_fn(x, x_mask, g=None, reverse=False):
        out, _ = m_rcl(
            to_time_last(x),
            to_time_last(x_mask),
            g=to_time_last(g) if g is not None else None,
            reverse=reverse,
        )
        return to_time_first(out)

    def torch_fn(x, x_mask, g=None, reverse=False):
        with torch.no_grad():
            out, _ = t_rcl(x, x_mask, g=g, reverse=reverse)
        return out

    return mlx_fn, torch_fn


class TestSynthesizerResidualCouplingLayerMeanOnly(BaseOperationTest):
    """`mean_only=True` coupling layer (the variant used inside ResidualCouplingBlock)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_residual_coupling_layer_pair(mean_only=True)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "rcl_mean_only")

        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 8, 16)).astype(np.float32)
        mask = np.ones((1, 1, 16), dtype=np.float32)
        cls.suite.add_test_case(
            name="rcl_mean_only_forward",
            inputs={"x": x, "x_mask": mask, "reverse": False},
            description="mean_only coupling layer in forward mode",
            atol=1e-4,
            rtol=1e-4,
        )
        cls.suite.add_test_case(
            name="rcl_mean_only_reverse",
            inputs={"x": x, "x_mask": mask, "reverse": True},
            description="mean_only coupling layer in reverse mode (the inference path)",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerResidualCouplingLayerWithG(BaseOperationTest):
    """Coupling layer with speaker conditioning through WN's cond path."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_residual_coupling_layer_pair(
            gin_channels=4, mean_only=True, seed=1
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "rcl_with_g")

        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 8, 12)).astype(np.float32)
        mask = np.ones((1, 1, 12), dtype=np.float32)
        g = rng.standard_normal((1, 4, 1)).astype(np.float32)
        cls.suite.add_test_case(
            name="rcl_with_g_reverse",
            inputs={"x": x, "x_mask": mask, "g": g, "reverse": True},
            description="Coupling layer with speaker conditioning in reverse mode",
            atol=1e-4,
            rtol=1e-4,
        )


def _build_residual_coupling_block_pair(
    channels=8, hidden_channels=16, kernel_size=5, dilation_rate=1, n_layers=2,
    n_flows=4, gin_channels=4, seed=0,
):
    torch.manual_seed(seed)
    init = dict(
        channels=channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers,
        n_flows=n_flows,
        gin_channels=gin_channels,
    )
    t_rcb = TorchResidualCouplingBlock(**init)
    # Randomize post weights inside each coupling layer (default zero-init makes the block a pure identity sequence
    # of Flips).
    with torch.no_grad():
        for flow in t_rcb.flows:
            if hasattr(flow, "post"):
                flow.post.weight.copy_(torch.randn_like(flow.post.weight) * 0.1)
                flow.post.bias.copy_(torch.randn_like(flow.post.bias) * 0.1)
    m_rcb = ResidualCouplingBlock(**init)
    copy_residual_coupling_block(t_rcb, m_rcb)
    set_eval(t_rcb, m_rcb)

    def mlx_fn(x, x_mask, g=None, reverse=False):
        out = m_rcb(
            to_time_last(x),
            to_time_last(x_mask),
            g=to_time_last(g) if g is not None else None,
            reverse=reverse,
        )
        return to_time_first(out)

    def torch_fn(x, x_mask, g=None, reverse=False):
        with torch.no_grad():
            return t_rcb(x, x_mask, g=g, reverse=reverse)

    return mlx_fn, torch_fn


class TestSynthesizerResidualCouplingBlockReverse(BaseOperationTest):
    """The inference path: ResidualCouplingBlock in reverse mode (the only mode RVC actually uses at inference)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_residual_coupling_block_pair()
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "rcb_reverse")

        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 8, 16)).astype(np.float32)
        mask = np.ones((1, 1, 16), dtype=np.float32)
        g = rng.standard_normal((1, 4, 1)).astype(np.float32)
        cls.suite.add_test_case(
            name="rcb_reverse_full",
            inputs={"x": x, "x_mask": mask, "g": g, "reverse": True},
            description="ResidualCouplingBlock reverse pass with 4 flows + speaker conditioning",
            atol=1e-4,
            rtol=1e-4,
        )


class TestSynthesizerResidualCouplingBlockForward(BaseOperationTest):
    """ResidualCouplingBlock in forward mode (training path), to round out coverage."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_residual_coupling_block_pair(seed=1)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "rcb_forward")

        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 8, 16)).astype(np.float32)
        mask = np.ones((1, 1, 16), dtype=np.float32)
        g = rng.standard_normal((1, 4, 1)).astype(np.float32)
        cls.suite.add_test_case(
            name="rcb_forward_full",
            inputs={"x": x, "x_mask": mask, "g": g, "reverse": False},
            description="ResidualCouplingBlock forward pass (training path)",
            atol=1e-4,
            rtol=1e-4,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
