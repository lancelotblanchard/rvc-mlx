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
    convert_rmvpe_checkpoint,
    convert_synthesizer_checkpoint,
    ensure_safetensors,
    ensure_synthesizer_safetensors,
    _fuse_weight_norm_state_dict,
    _normalize_synth_config,
    _SYNTH_CONFIG_KEYS,
)
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

        np.testing.assert_allclose(mlx_o_np, torch_o_np, atol=5e-3, rtol=5e-3)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
