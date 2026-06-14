"""
Paired-module tests for the HuBERT / ContentVec content encoder.

Each test builds a PyTorch reference and an MLX implementation with matching hyperparameters and copies weights
across, then compares forward outputs. The PyTorch side uses channels-first conventions for Conv1d / GroupNorm; the
MLX side runs channels-last natively. Test wrappers transpose at the boundary.
"""

import numpy as np
import pytest
import torch

from rvc_mlx.hubert import (
    FeatureExtractor,
    HUBERT_BASE_CONV_LAYERS,
    HubertModel,
    HubertMultiHeadAttention,
    HubertTransformerEncoder,
    PositionalConv,
    TransformerSentenceEncoderLayer,
)
from rvc_mlx._torch_ref import (
    TorchFeatureExtractor,
    TorchHubertModel,
    TorchHubertMultiHeadAttention,
    TorchHubertTransformerEncoder,
    TorchPositionalConv,
    TorchTransformerSentenceEncoderLayer,
)

from .mlx_torch_comparison_framework import BaseOperationTest, OperationTestSuite
from .torch_bridge import (
    copy_feature_extractor,
    copy_hubert_model,
    copy_hubert_multi_head_attention,
    copy_hubert_transformer_encoder,
    copy_positional_conv,
    copy_transformer_sentence_encoder_layer,
    set_eval,
)


# ----------------------------------------------------------------------------------------------------------------------
# FeatureExtractor.
# ----------------------------------------------------------------------------------------------------------------------


# Scaled-down conv stack for tests. Still exercises:
#   * the layer-0 GroupNorm branch
#   * multiple subsequent conv layers with strides
# Real HuBERT-base downsamples 320x; this config downsamples 5 * 2 = 10x to keep test inputs tiny.
_SMALL_CONV_LAYERS = (
    (16, 10, 5),
    (16, 3, 2),
)


def _build_feature_extractor_pair(conv_layers=_SMALL_CONV_LAYERS, seed=0):
    torch.manual_seed(seed)
    t_fe = TorchFeatureExtractor(conv_layers)
    m_fe = FeatureExtractor(conv_layers)
    copy_feature_extractor(t_fe, m_fe)
    set_eval(t_fe, m_fe)

    def mlx_fn(audio):
        # audio: (B, T_audio) -> MLX output (B, T_out, C); transpose to (B, C, T_out) for comparison.
        import mlx.core as mx

        out = m_fe(audio)
        return mx.transpose(out, (0, 2, 1))

    def torch_fn(audio):
        with torch.no_grad():
            return t_fe(audio)

    return mlx_fn, torch_fn


class TestHubertFeatureExtractor(BaseOperationTest):
    """Exercises layer-0 GroupNorm + subsequent strided convs."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_feature_extractor_pair()
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_feature_extractor")

        rng = np.random.default_rng(0)
        cls.suite.add_test_case(
            name="feature_extractor_small",
            inputs={"audio": rng.standard_normal((1, 320)).astype(np.float32)},
            description="320 audio samples through 2-layer small extractor (10x downsample)",
            atol=1e-4,
            rtol=1e-4,
        )
        cls.suite.add_test_case(
            name="feature_extractor_batched",
            inputs={"audio": rng.standard_normal((2, 480)).astype(np.float32)},
            description="Batched audio through the small extractor",
            atol=1e-4,
            rtol=1e-4,
        )


# ----------------------------------------------------------------------------------------------------------------------
# HubertMultiHeadAttention (standard scaled dot-product attention with Linear Q/K/V/O).
# ----------------------------------------------------------------------------------------------------------------------


def _build_hubert_mha_pair(embed_dim=64, num_heads=4, seed=0):
    torch.manual_seed(seed)
    t_mha = TorchHubertMultiHeadAttention(embed_dim, num_heads)
    m_mha = HubertMultiHeadAttention(embed_dim, num_heads)
    copy_hubert_multi_head_attention(t_mha, m_mha)
    set_eval(t_mha, m_mha)

    def mlx_fn(x, key_padding_mask=None):
        return m_mha(x, key_padding_mask=key_padding_mask)

    def torch_fn(x, key_padding_mask=None):
        with torch.no_grad():
            return t_mha(x, key_padding_mask=key_padding_mask)

    return mlx_fn, torch_fn


class TestHubertMHANoMask(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_hubert_mha_pair()
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_mha_no_mask")

        rng = np.random.default_rng(0)
        cls.suite.add_test_case(
            name="mha_no_mask",
            inputs={"x": rng.standard_normal((1, 16, 64)).astype(np.float32)},
            description="HuBERT self-attention without a key-padding mask",
            atol=1e-4,
            rtol=1e-4,
        )


class TestHubertMHAWithMask(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_hubert_mha_pair(seed=1)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_mha_with_mask")

        rng = np.random.default_rng(1)
        x = rng.standard_normal((2, 12, 64)).astype(np.float32)
        # Mark the last 4 positions of the second batch entry as invalid.
        mask = np.zeros((2, 12), dtype=bool)
        mask[1, 8:] = True
        cls.suite.add_test_case(
            name="mha_partial_mask",
            inputs={"x": x, "key_padding_mask": mask},
            description="Self-attention with the last 4 frames of batch entry 1 padding-masked",
            atol=1e-4,
            rtol=1e-4,
        )


# ----------------------------------------------------------------------------------------------------------------------
# TransformerSentenceEncoderLayer (pre-norm + GELU FFN).
# ----------------------------------------------------------------------------------------------------------------------


def _build_transformer_layer_pair(embed_dim=64, ffn_dim=128, num_heads=4, seed=0):
    torch.manual_seed(seed)
    t_layer = TorchTransformerSentenceEncoderLayer(embed_dim, ffn_dim, num_heads)
    m_layer = TransformerSentenceEncoderLayer(embed_dim, ffn_dim, num_heads)
    copy_transformer_sentence_encoder_layer(t_layer, m_layer)
    set_eval(t_layer, m_layer)

    def mlx_fn(x, key_padding_mask=None):
        return m_layer(x, key_padding_mask=key_padding_mask)

    def torch_fn(x, key_padding_mask=None):
        with torch.no_grad():
            return t_layer(x, key_padding_mask=key_padding_mask)

    return mlx_fn, torch_fn


class TestHubertTransformerLayer(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_transformer_layer_pair()
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_transformer_layer")

        rng = np.random.default_rng(0)
        cls.suite.add_test_case(
            name="layer_no_mask",
            inputs={"x": rng.standard_normal((1, 16, 64)).astype(np.float32)},
            description="Single pre-norm transformer block with GELU FFN",
            atol=1e-4,
            rtol=1e-4,
        )


# ----------------------------------------------------------------------------------------------------------------------
# PositionalConv (grouped Conv1d + SamePad + GELU).
# ----------------------------------------------------------------------------------------------------------------------


def _build_positional_conv_pair(embed_dim=64, kernel_size=8, groups=4, seed=0):
    torch.manual_seed(seed)
    t_pc = TorchPositionalConv(embed_dim, kernel_size=kernel_size, groups=groups)
    m_pc = PositionalConv(embed_dim, kernel_size=kernel_size, groups=groups)
    copy_positional_conv(t_pc, m_pc)
    set_eval(t_pc, m_pc)

    def mlx_fn(x):
        return m_pc(x)

    def torch_fn(x):
        with torch.no_grad():
            return t_pc(x)

    return mlx_fn, torch_fn


class TestHubertPositionalConvEvenKernel(BaseOperationTest):
    """Even kernel size triggers the SamePad crop."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_positional_conv_pair(kernel_size=8)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_pos_conv_even")

        rng = np.random.default_rng(0)
        cls.suite.add_test_case(
            name="pos_conv_even",
            inputs={"x": rng.standard_normal((1, 16, 64)).astype(np.float32)},
            description="kernel=8 -> conv outputs T+1 frames -> SamePad crops to T",
            atol=1e-4,
            rtol=1e-4,
        )


class TestHubertPositionalConvOddKernel(BaseOperationTest):
    """Odd kernel size: SamePad is a no-op."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_positional_conv_pair(kernel_size=7, seed=1)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_pos_conv_odd")

        rng = np.random.default_rng(1)
        cls.suite.add_test_case(
            name="pos_conv_odd",
            inputs={"x": rng.standard_normal((1, 16, 64)).astype(np.float32)},
            description="kernel=7 (odd) -> conv preserves length, SamePad is identity",
            atol=1e-4,
            rtol=1e-4,
        )


# ----------------------------------------------------------------------------------------------------------------------
# HubertTransformerEncoder (full encoder with pos_conv + LayerNorm + N layers).
# ----------------------------------------------------------------------------------------------------------------------


def _build_hubert_encoder_pair(
    embed_dim=64, ffn_dim=128, num_layers=2, num_heads=4, pos_conv_kernel=8, pos_conv_groups=4, seed=0
):
    torch.manual_seed(seed)
    init = dict(
        embed_dim=embed_dim,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        pos_conv_kernel=pos_conv_kernel,
        pos_conv_groups=pos_conv_groups,
    )
    t_enc = TorchHubertTransformerEncoder(**init)
    m_enc = HubertTransformerEncoder(**init)
    copy_hubert_transformer_encoder(t_enc, m_enc)
    set_eval(t_enc, m_enc)

    def mlx_fn(x, output_layer=None):
        out, _ = m_enc(x, output_layer=output_layer)
        return out

    def torch_fn(x, output_layer=None):
        with torch.no_grad():
            out, _ = t_enc(x, output_layer=output_layer)
        return out

    return mlx_fn, torch_fn


class TestHubertEncoderFull(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_hubert_encoder_pair()
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_encoder_full")

        rng = np.random.default_rng(0)
        cls.suite.add_test_case(
            name="encoder_full_depth",
            inputs={"x": rng.standard_normal((1, 16, 64)).astype(np.float32)},
            description="2-layer encoder, full-depth output",
            atol=1e-3,
            rtol=1e-3,
        )


class TestHubertEncoderEarlyExit(BaseOperationTest):
    """Exercising the `output_layer` early-exit path (RVC uses it to pull layer 9 or 12)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_hubert_encoder_pair(num_layers=4, seed=1)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_encoder_early")

        rng = np.random.default_rng(1)
        cls.suite.add_test_case(
            name="encoder_layer_2_of_4",
            inputs={"x": rng.standard_normal((1, 12, 64)).astype(np.float32), "output_layer": 2},
            description="4-layer encoder, stop after layer 2 (output_layer=2)",
            atol=1e-3,
            rtol=1e-3,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Full HubertModel.
# ----------------------------------------------------------------------------------------------------------------------


# Scaled-down config for the end-to-end test. Still exercises every code path (feature extractor with GroupNorm,
# layer_norm, post_extract_proj, encoder with pos_conv + multiple transformer layers, optional final_proj).
_SMALL_HUBERT_CONFIG = dict(
    conv_layers=_SMALL_CONV_LAYERS,
    extractor_mode="default",
    embed_dim=64,
    encoder_ffn_dim=128,
    encoder_layers=2,
    encoder_attention_heads=4,
    pos_conv_kernel=8,
    pos_conv_groups=4,
    has_final_proj=False,
)


def _build_hubert_model_pair(seed=0, **overrides):
    config = {**_SMALL_HUBERT_CONFIG, **overrides}
    torch.manual_seed(seed)
    t_hm = TorchHubertModel(**config)
    m_hm = HubertModel(**config)
    copy_hubert_model(t_hm, m_hm)
    set_eval(t_hm, m_hm)
    return t_hm, m_hm


class TestHubertModelFullDepth(BaseOperationTest):
    """Extracting at the last transformer layer (output_layer=None)."""

    @classmethod
    def setup_class(cls):
        t_hm, m_hm = _build_hubert_model_pair()

        def mlx_fn(audio):
            return m_hm.extract_features(audio)

        def torch_fn(audio):
            with torch.no_grad():
                return t_hm.extract_features(audio)

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_model_full")

        rng = np.random.default_rng(0)
        # Use enough samples so the conv stack produces several output frames.
        cls.suite.add_test_case(
            name="hubert_extract_full",
            inputs={"audio": rng.standard_normal((1, 320)).astype(np.float32)},
            description="HuBERT extract_features end-to-end with default (last-layer) output",
            atol=5e-3,
            rtol=5e-3,
        )


class TestHubertModelEarlyExit(BaseOperationTest):
    """Extracting at a specific intermediate layer (RVC v1 uses layer 9 of HuBERT-base; here we test layer 1 of 2)."""

    @classmethod
    def setup_class(cls):
        t_hm, m_hm = _build_hubert_model_pair(seed=1)

        def mlx_fn(audio, output_layer):
            return m_hm.extract_features(audio, output_layer=output_layer)

        def torch_fn(audio, output_layer):
            with torch.no_grad():
                return t_hm.extract_features(audio, output_layer=output_layer)

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_model_early")

        rng = np.random.default_rng(1)
        cls.suite.add_test_case(
            name="hubert_extract_layer_1",
            inputs={"audio": rng.standard_normal((1, 480)).astype(np.float32), "output_layer": 1},
            description="extract_features with output_layer=1 stops after the first transformer block",
            atol=5e-3,
            rtol=5e-3,
        )


class TestHubertModelWithFinalProj(BaseOperationTest):
    """HuBERT-v1 / non-ContentVec checkpoint variant with `has_final_proj=True`."""

    @classmethod
    def setup_class(cls):
        t_hm, m_hm = _build_hubert_model_pair(seed=2, has_final_proj=True)

        def mlx_fn(audio):
            return m_hm.extract_features(audio)

        def torch_fn(audio):
            with torch.no_grad():
                return t_hm.extract_features(audio)

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "hubert_model_final_proj")

        rng = np.random.default_rng(2)
        cls.suite.add_test_case(
            name="hubert_with_final_proj",
            inputs={"audio": rng.standard_normal((1, 320)).astype(np.float32)},
            description="extract_features with the final_proj Linear applied",
            atol=5e-3,
            rtol=5e-3,
        )


def test_hubert_base_conv_layers_constant():
    """Sanity check: the published HuBERT-base downsamples 320x."""
    total = 1
    for _, _, s in HUBERT_BASE_CONV_LAYERS:
        total *= s
    assert total == 320


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
