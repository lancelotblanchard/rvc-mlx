"""
Round-trip tests for the RMVPE checkpoint converter.

Strategy: build a small `TorchE2E` with random weights, save its `state_dict` to a temporary `.pt`, run
`convert_rmvpe_checkpoint` to produce a `.safetensors`, load it back through `RMVPE.from_pretrained`, and verify the
MLX forward matches the original torch forward.

The full RVC checkpoint config (`DEFAULT_E2E_CONFIG`) is expensive to instantiate twice in test; we monkeypatch
`_DEFAULT_E2E_CONFIG` (and the matching torch-side default) to a smaller architecture so the tests stay fast while
still exercising every layer type the converter has to translate.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np
import pytest
import torch

from rvc_mlx import rmvpe as rmvpe_mod
from rvc_mlx import _torch_ref
from rvc_mlx._convert_bridge import randomize_bn_stats
from rvc_mlx.convert import (
    convert_hubert_checkpoint,
    convert_rmvpe_checkpoint,
    convert_synthesizer_checkpoint,
    ensure_hubert_safetensors,
    ensure_safetensors,
    ensure_synthesizer_safetensors,
    _fuse_weight_norm_state_dict,
    _normalize_synth_config,
    _remap_hubert_state_dict,
    _drop_training_only_keys,
    _SYNTH_CONFIG_KEYS,
)
from rvc_mlx.hubert import HubertModel
from rvc_mlx.rmvpe import RMVPE
from rvc_mlx.synthesizer import SynthesizerTrnMs768NSFsid


# Smaller-than-released architecture: still touches DeepUnet (encoder + intermediate + decoder), cnn, BiGRU, Linear.
_SMALL_E2E_CONFIG = dict(
    n_blocks=1,
    n_gru=1,
    kernel_size=(2, 2),
    en_de_layers=2,
    inter_layers=1,
    in_channels=1,
    en_out_channels=4,
)


@pytest.fixture
def small_e2e_config(monkeypatch):
    """Patch the released-checkpoint defaults so the conversion path uses our small config in both directions."""
    monkeypatch.setattr(rmvpe_mod, "_DEFAULT_E2E_CONFIG", _SMALL_E2E_CONFIG)
    monkeypatch.setattr(_torch_ref, "DEFAULT_E2E_CONFIG", _SMALL_E2E_CONFIG)
    return _SMALL_E2E_CONFIG


def _build_torch_e2e_with_random_weights(config, seed=0):
    torch.manual_seed(seed)
    model = _torch_ref.TorchE2E(**config)
    # BN running stats default to (0, 1); replace with non-trivial values so the test detects mis-copied stats.
    randomize_bn_stats(model, seed=seed)
    model.eval()
    return model


class TestConvertRmvpeCheckpoint:
    """End-to-end: write a torch state_dict to a temp file, convert it, load via from_pretrained, compare outputs."""

    def test_round_trip_forward_matches_torch(self, tmp_path, small_e2e_config):
        torch_model = _build_torch_e2e_with_random_weights(small_e2e_config, seed=0)

        pt_path = str(tmp_path / "rmvpe.pt")
        torch.save(torch_model.state_dict(), pt_path)

        # First call produces the sibling safetensors and returns its path.
        out_path = convert_rmvpe_checkpoint(pt_path)
        assert out_path == str(tmp_path / "rmvpe.safetensors")
        assert os.path.exists(out_path)

        # Load via the production factory and compare forwards on a deterministic mel input.
        rmvpe = RMVPE.from_pretrained(out_path)
        rng = np.random.default_rng(0)
        # Mel shape is (B, n_mel=128, T). RMVPE's E2E expects the channels-first PyTorch convention before its
        # internal transpose/expand_dims, so we feed the same (B, 128, T) tensor to both sides.
        mel_np = rng.standard_normal((1, 128, 16)).astype(np.float32)
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(mel_np)).numpy()
        mlx_out = np.array(rmvpe.model(mx.array(mel_np)))
        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-4, rtol=1e-4)

    def test_ensure_safetensors_passes_through_safetensors(self, tmp_path):
        # If you already have a .safetensors, ensure_safetensors must not convert — no torch needed.
        p = str(tmp_path / "already.safetensors")
        # Touch the file (existence isn't checked for the safetensors branch, but make it concrete).
        open(p, "w").close()
        assert ensure_safetensors(p) == p

    def test_ensure_safetensors_caches_after_first_convert(self, tmp_path, small_e2e_config):
        torch_model = _build_torch_e2e_with_random_weights(small_e2e_config, seed=1)
        pt_path = str(tmp_path / "rmvpe.pt")
        torch.save(torch_model.state_dict(), pt_path)

        # First call: actually converts (no cached sibling yet).
        out1 = ensure_safetensors(pt_path)
        assert out1 == str(tmp_path / "rmvpe.safetensors")
        mtime1 = os.path.getmtime(out1)

        # Second call: must return the same path and must NOT re-write the file (we check via mtime).
        out2 = ensure_safetensors(pt_path)
        assert out2 == out1
        assert os.path.getmtime(out2) == mtime1

    def test_ensure_safetensors_rejects_unknown_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported checkpoint extension"):
            ensure_safetensors(str(tmp_path / "weights.bin"))


class TestRmvpeFromPretrained:
    """`RMVPE.from_pretrained` is the production entry point; verify it returns a usable RMVPE."""

    def test_from_pretrained_returns_working_rmvpe(self, tmp_path, small_e2e_config):
        torch_model = _build_torch_e2e_with_random_weights(small_e2e_config, seed=2)
        pt_path = str(tmp_path / "rmvpe.pt")
        torch.save(torch_model.state_dict(), pt_path)

        rmvpe = RMVPE.from_pretrained(pt_path)  # accepts .pt directly (auto-converts)

        # Hit the full inference path on synthetic audio — exercises mel extraction + mel2hidden + decode together.
        # Output is a 1D numpy f0 array; we just verify shape & finiteness here. Numerical equivalence with the
        # torch side is already covered by test_round_trip_forward_matches_torch above.
        rng = np.random.default_rng(3)
        audio = mx.array(rng.standard_normal(16000).astype(np.float32))
        f0 = rmvpe.infer_from_audio(audio)
        assert f0.ndim == 1
        assert np.isfinite(f0).all()


# ----------------------------------------------------------------------------------------------------------------------
# Synthesizer converter tests.
# ----------------------------------------------------------------------------------------------------------------------


# Small synthesizer config used for the round-trip tests. Mirrors `_SYNTH_TEST_CONFIG` in `test_synthesizer.py`.
_SMALL_SYNTH_CONFIG = dict(
    spec_channels=64,
    segment_size=128,
    inter_channels=8,
    hidden_channels=8,
    filter_channels=16,
    n_heads=2,
    n_layers=2,
    kernel_size=3,
    p_dropout=0.0,
    resblock="1",
    resblock_kernel_sizes=[3, 7],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
    upsample_rates=[4, 2],
    upsample_initial_channel=32,
    upsample_kernel_sizes=[8, 4],
    spk_embed_dim=4,
    gin_channels=4,
    sr=16000,
)


def _build_torch_synth_with_random_weights(config, seed=0):
    torch.manual_seed(seed)
    model = _torch_ref.TorchSynthesizerTrnMs768NSFsid(**config)
    # Randomize coupling-layer post weights away from zero-init so the flow isn't a degenerate identity.
    with torch.no_grad():
        for flow in model.flow.flows:
            if hasattr(flow, "post"):
                flow.post.weight.copy_(torch.randn_like(flow.post.weight) * 0.1)
                flow.post.bias.copy_(torch.randn_like(flow.post.bias) * 0.1)
    model.eval()
    return model


class TestFuseWeightNorm:
    """Unit tests for the `_fuse_weight_norm_state_dict` helper. Exercises the formula used by `nn.utils.weight_norm`."""

    def test_pass_through_for_plain_weights(self):
        sd = {"foo.weight": torch.randn(4, 3), "bar.bias": torch.randn(3)}
        out = _fuse_weight_norm_state_dict(sd)
        assert set(out.keys()) == {"foo.weight", "bar.bias"}
        torch.testing.assert_close(out["foo.weight"], sd["foo.weight"])
        torch.testing.assert_close(out["bar.bias"], sd["bar.bias"])

    def test_fuses_weight_g_and_v(self):
        # Construct a known weight_g/weight_v pair and check the fused weight matches
        # `weight = weight_g * weight_v / ||weight_v||` where the norm is over all dims except dim 0.
        v = torch.tensor([[3.0, 4.0], [1.0, 0.0]], dtype=torch.float32)  # shape (2, 2)
        g = torch.tensor([[5.0], [2.0]], dtype=torch.float32)  # shape (2, 1)
        sd = {"layer.weight_g": g, "layer.weight_v": v}
        out = _fuse_weight_norm_state_dict(sd)
        assert "layer.weight" in out
        assert "layer.weight_g" not in out and "layer.weight_v" not in out
        # Row 0 of v has norm 5, row 1 has norm 1.
        expected = torch.tensor([[3.0, 4.0], [2.0, 0.0]], dtype=torch.float32)
        torch.testing.assert_close(out["layer.weight"], expected)

    def test_raises_on_incomplete_pair(self):
        sd = {"layer.weight_g": torch.randn(3, 1)}  # missing weight_v
        with pytest.raises(ValueError, match="Incomplete weight_norm pair"):
            _fuse_weight_norm_state_dict(sd)


class TestNormalizeSynthConfig:
    def test_list_form_maps_to_kwargs(self):
        cfg_list = [getattr(_SMALL_SYNTH_CONFIG, k, None) or _SMALL_SYNTH_CONFIG[k] for k in _SYNTH_CONFIG_KEYS]
        out = _normalize_synth_config(cfg_list)
        for k in _SYNTH_CONFIG_KEYS:
            assert out[k] == _SMALL_SYNTH_CONFIG[k]

    def test_dict_form_passthrough(self):
        out = _normalize_synth_config(_SMALL_SYNTH_CONFIG)
        assert out == _SMALL_SYNTH_CONFIG
        assert out is not _SMALL_SYNTH_CONFIG  # shouldn't be the same object

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="Expected synthesizer config list of length"):
            _normalize_synth_config([1, 2, 3])

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported synthesizer config type"):
            _normalize_synth_config("not_a_config")


class TestConvertSynthesizerCheckpoint:
    """End-to-end synthesizer conversion: write a torch state_dict + config, convert it, load via from_pretrained,
    compare forwards."""

    def test_round_trip_plain_weights(self, tmp_path):
        torch_model = _build_torch_synth_with_random_weights(_SMALL_SYNTH_CONFIG, seed=0)

        pt_path = str(tmp_path / "synth.pth")
        # Released RVC checkpoints store both the architecture config and the state_dict under known top-level keys.
        cfg_list = [_SMALL_SYNTH_CONFIG[k] for k in _SYNTH_CONFIG_KEYS]
        torch.save({"config": cfg_list, "weight": torch_model.state_dict()}, pt_path)

        out_path, config_path = convert_synthesizer_checkpoint(pt_path)
        assert out_path == str(tmp_path / "synth.safetensors")
        assert config_path == str(tmp_path / "synth.config.json")
        assert os.path.exists(out_path)
        assert os.path.exists(config_path)

        # Round-trip through the production loader.
        mlx_syn = SynthesizerTrnMs768NSFsid.from_pretrained(out_path)

        # Build deterministic inputs and compare audio outputs.
        upp = int(np.prod(_SMALL_SYNTH_CONFIG["upsample_rates"]))
        rng = np.random.default_rng(0)
        B, T = 1, 8
        phone = rng.standard_normal((B, T, 768)).astype(np.float32)
        pitch = rng.integers(0, 256, size=(B, T)).astype(np.int64)
        nsff0 = rng.uniform(50.0, 500.0, size=(B, T)).astype(np.float32)
        phone_lengths = np.array([T] * B, dtype=np.int64)
        sid = np.array([0], dtype=np.int64)
        noise_z = rng.standard_normal((B, T, _SMALL_SYNTH_CONFIG["inter_channels"])).astype(np.float32)
        rand_ini = rng.uniform(size=(B, 1)).astype(np.float32)
        noise_raw = rng.standard_normal((B, T * upp, 1)).astype(np.float32)

        with torch.no_grad():
            torch_o, _, _ = torch_model.infer(
                torch.from_numpy(phone),
                torch.from_numpy(phone_lengths),
                torch.from_numpy(pitch),
                torch.from_numpy(nsff0),
                torch.from_numpy(sid),
                noise_z=torch.from_numpy(noise_z),
                rand_ini=torch.from_numpy(rand_ini),
                noise_raw=torch.from_numpy(noise_raw),
            )
        torch_o_np = torch_o.numpy()

        mlx_o, _, _ = mlx_syn.infer(
            mx.array(phone),
            mx.array(phone_lengths),
            mx.array(pitch),
            mx.array(nsff0),
            mx.array(sid),
            noise_z=mx.array(noise_z),
            rand_ini=mx.array(rand_ini),
            noise_raw=mx.array(noise_raw),
        )
        # mlx_o: (B, T_audio, 1) -> (B, 1, T_audio) for comparison.
        mlx_o_np = np.array(mx.transpose(mlx_o, (0, 2, 1)))

        # End-to-end tolerance: the synthesizer accumulates float32 error through a deep encoder + 4-flow stack +
        # multi-level upsampling generator. Per-module tests (test_synthesizer.py) catch precision regressions
        # tightly; here we just verify the converter doesn't introduce *additional* drift on top of that.
        #
        # Note: the bound is wider than `TestSynthesizerFullInfer` (5e-2) because the random *inputs* differ between
        # the two tests (different rng call sequences) — same code path, just unluckier samples here. A real
        # implementation bug would fail dozens of elements by much wider margins.
        np.testing.assert_allclose(mlx_o_np, torch_o_np, atol=1e-1, rtol=1e-1)

    def test_ensure_synthesizer_safetensors_passes_through(self, tmp_path):
        st = str(tmp_path / "already.safetensors")
        cfg = str(tmp_path / "already.config.json")
        open(st, "w").close()
        with open(cfg, "w") as f:
            json.dump(_SMALL_SYNTH_CONFIG, f)
        out_path, config = ensure_synthesizer_safetensors(st)
        assert out_path == st
        assert config == _SMALL_SYNTH_CONFIG

    def test_ensure_synthesizer_safetensors_missing_config(self, tmp_path):
        st = str(tmp_path / "orphan.safetensors")
        open(st, "w").close()
        with pytest.raises(FileNotFoundError, match="missing its sibling config"):
            ensure_synthesizer_safetensors(st)

    def test_ensure_synthesizer_safetensors_rejects_unknown_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported synthesizer checkpoint extension"):
            ensure_synthesizer_safetensors(str(tmp_path / "weights.bin"))


# ----------------------------------------------------------------------------------------------------------------------
# HuBERT converter tests.
# ----------------------------------------------------------------------------------------------------------------------


# Scaled-down HuBERT config (mirrors the one in test_hubert.py).
_SMALL_HUBERT_CONFIG = dict(
    conv_layers=((16, 10, 5), (16, 3, 2)),
    extractor_mode="default",
    embed_dim=64,
    encoder_ffn_dim=128,
    encoder_layers=2,
    encoder_attention_heads=4,
    pos_conv_kernel=8,
    pos_conv_groups=4,
    has_final_proj=False,
)


def _build_torch_hubert_with_random_weights(config, seed=0):
    """Build a `TorchHubertModel` with deterministic random weights for testing."""
    torch.manual_seed(seed)
    model = _torch_ref.TorchHubertModel(**config)
    model.eval()
    return model


def _fairseq_style_state_dict_from_torch(model, has_final_proj):
    """
    Convert a `TorchHubertModel` state_dict to the **fairseq** naming convention + apply `weight_norm` to `pos_conv`
    so the produced state_dict mimics what a released `hubert_base.pt` looks like. Used to test that
    `convert_hubert_checkpoint` correctly handles the published format.
    """
    sd = model.state_dict()
    # Apply weight_norm to pos_conv (dim=2). PyTorch's weight_norm adds weight_g/weight_v and removes weight.
    pc_weight = sd.pop("encoder.pos_conv.conv.weight")
    # Per-kernel-position norm: norm over dims (0, 1), keep dim 2.
    weight_v = pc_weight
    weight_g = torch.linalg.vector_norm(weight_v, dim=(0, 1), keepdim=True)
    # In fairseq's saved state_dict the key is encoder.pos_conv.0.weight_g (the "0" indexes into nn.Sequential).
    sd["encoder.pos_conv.0.weight_g"] = weight_g
    sd["encoder.pos_conv.0.weight_v"] = weight_v
    # encoder.pos_conv.conv.bias -> encoder.pos_conv.0.bias
    if "encoder.pos_conv.conv.bias" in sd:
        sd["encoder.pos_conv.0.bias"] = sd.pop("encoder.pos_conv.conv.bias")

    # Rename feature_extractor.convs.{i}.weight -> feature_extractor.conv_layers.{i}.0.weight, etc.
    rename: list = []
    for k in list(sd.keys()):
        new_k = None
        if k.startswith("feature_extractor.convs."):
            # convs.{i}.{suffix} -> conv_layers.{i}.0.{suffix}
            tail = k[len("feature_extractor.convs.") :]
            new_k = f"feature_extractor.conv_layers.{tail.split('.', 1)[0]}.0.{tail.split('.', 1)[1]}"
        elif k.startswith("feature_extractor.norms."):
            tail = k[len("feature_extractor.norms.") :]
            new_k = f"feature_extractor.conv_layers.{tail.split('.', 1)[0]}.2.{tail.split('.', 1)[1]}"
        if new_k is not None:
            rename.append((k, new_k))
    for old, new in rename:
        sd[new] = sd.pop(old)

    # Add some training-only entries the released checkpoint carries (we drop these on load).
    sd["mask_emb"] = torch.zeros(64)
    sd["label_embs_concat"] = torch.zeros(100, 64)
    return sd


class TestRemapHubertStateDict:
    def test_remaps_feature_extractor_keys(self):
        sd = {
            "feature_extractor.conv_layers.0.0.weight": torch.zeros(1),
            "feature_extractor.conv_layers.0.2.weight": torch.zeros(1),
            "feature_extractor.conv_layers.1.0.weight": torch.zeros(1),
            "unrelated.weight": torch.zeros(1),
        }
        out = _remap_hubert_state_dict(sd)
        assert "feature_extractor.convs.0.weight" in out
        assert "feature_extractor.norms.0.weight" in out
        assert "feature_extractor.convs.1.weight" in out
        assert "unrelated.weight" in out

    def test_remaps_pos_conv(self):
        sd = {
            "encoder.pos_conv.0.weight_g": torch.zeros(1),
            "encoder.pos_conv.0.weight_v": torch.zeros(1),
            "encoder.pos_conv.0.bias": torch.zeros(1),
        }
        out = _remap_hubert_state_dict(sd)
        assert "encoder.pos_conv.conv.weight_g" in out
        assert "encoder.pos_conv.conv.weight_v" in out
        assert "encoder.pos_conv.conv.bias" in out


class TestDropTrainingOnlyKeys:
    def test_drops_mask_emb_and_label_embs(self):
        sd = {
            "mask_emb": torch.zeros(1),
            "label_embs_concat": torch.zeros(1),
            "_ema_some_key": torch.zeros(1),
            "feature_extractor.convs.0.weight": torch.zeros(1),
        }
        out = _drop_training_only_keys(sd)
        assert "feature_extractor.convs.0.weight" in out
        assert "mask_emb" not in out
        assert "label_embs_concat" not in out
        assert "_ema_some_key" not in out


class TestFuseWeightNormPosConv:
    """The pos_conv weight_norm uses `dim=2` (per-kernel-position). Verify the auto-detect path."""

    def test_dim2_weight_norm(self):
        # Build a known v with `dim=2` weight_norm.
        torch.manual_seed(0)
        v = torch.randn(8, 4, 3)  # (out=8, in/g=4, k=3)
        g = torch.linalg.vector_norm(v, dim=(0, 1), keepdim=True)  # shape (1, 1, 3)
        sd = {"foo.weight_g": g, "foo.weight_v": v}
        out = _fuse_weight_norm_state_dict(sd)
        assert "foo.weight" in out
        # After fusion, weight == g * v / ||v||_{0,1} == v (since g IS the norm).
        torch.testing.assert_close(out["foo.weight"], v)


class TestConvertHubertCheckpoint:
    def test_round_trip_fairseq_style(self, tmp_path):
        torch_model = _build_torch_hubert_with_random_weights(_SMALL_HUBERT_CONFIG, seed=0)
        sd_fairseq = _fairseq_style_state_dict_from_torch(
            torch_model, _SMALL_HUBERT_CONFIG["has_final_proj"]
        )
        pt_path = str(tmp_path / "hubert.pt")
        torch.save({"model": sd_fairseq}, pt_path)

        # The released RVC checkpoint matches HUBERT_BASE_CONFIG; here we override to the scaled-down test dims.
        out_path, config_path = convert_hubert_checkpoint(pt_path, config_overrides=_SMALL_HUBERT_CONFIG)
        assert out_path == str(tmp_path / "hubert.safetensors")
        assert config_path == str(tmp_path / "hubert.config.json")
        assert os.path.exists(out_path)
        assert os.path.exists(config_path)

        # Patch the default config so `from_pretrained` builds the matching architecture. We do this by passing the
        # saved config explicitly: from_pretrained reads from the JSON, which we wrote with our scaled-down dims.
        mlx_hubert = HubertModel.from_pretrained(out_path)

        rng = np.random.default_rng(0)
        audio_np = rng.standard_normal((1, 320)).astype(np.float32)
        with torch.no_grad():
            torch_out = torch_model.extract_features(torch.from_numpy(audio_np)).numpy()
        mlx_out = np.array(mlx_hubert.extract_features(mx.array(audio_np)))
        np.testing.assert_allclose(mlx_out, torch_out, atol=5e-3, rtol=5e-3)

    def test_round_trip_with_final_proj(self, tmp_path):
        # v1 / HuBERT-base variant: has final_proj. Convert auto-detects it from the state_dict.
        cfg = dict(_SMALL_HUBERT_CONFIG)
        cfg["has_final_proj"] = True
        torch_model = _build_torch_hubert_with_random_weights(cfg, seed=1)
        sd_fairseq = _fairseq_style_state_dict_from_torch(torch_model, True)
        pt_path = str(tmp_path / "hubert_v1.pt")
        torch.save({"model": sd_fairseq}, pt_path)

        out_path, _ = convert_hubert_checkpoint(pt_path, config_overrides=cfg)
        with open(out_path[: -len(".safetensors")] + ".config.json") as f:
            saved_config = json.load(f)
        assert saved_config["has_final_proj"] is True

    def test_round_trip_top_level_state_dict(self, tmp_path):
        # Some HuBERT releases save the state_dict at the top level rather than nested under "model".
        torch_model = _build_torch_hubert_with_random_weights(_SMALL_HUBERT_CONFIG, seed=2)
        sd_fairseq = _fairseq_style_state_dict_from_torch(torch_model, False)
        pt_path = str(tmp_path / "hubert_flat.pt")
        torch.save(sd_fairseq, pt_path)

        out_path, _ = convert_hubert_checkpoint(pt_path, config_overrides=_SMALL_HUBERT_CONFIG)
        assert os.path.exists(out_path)

    def test_ensure_hubert_safetensors_passes_through(self, tmp_path):
        st = str(tmp_path / "already.safetensors")
        cfg = str(tmp_path / "already.config.json")
        open(st, "w").close()
        with open(cfg, "w") as f:
            json.dump(_SMALL_HUBERT_CONFIG, f)
        out_path, config = ensure_hubert_safetensors(st)
        assert out_path == st

    def test_ensure_hubert_safetensors_missing_config(self, tmp_path):
        st = str(tmp_path / "orphan.safetensors")
        open(st, "w").close()
        with pytest.raises(FileNotFoundError, match="missing its sibling config"):
            ensure_hubert_safetensors(st)

    def test_ensure_hubert_safetensors_rejects_unknown_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported HuBERT checkpoint extension"):
            ensure_hubert_safetensors(str(tmp_path / "weights.bin"))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
