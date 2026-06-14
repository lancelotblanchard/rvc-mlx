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

import os

import mlx.core as mx
import numpy as np
import pytest
import torch

from rvc_mlx import rmvpe as rmvpe_mod
from rvc_mlx import _torch_ref
from rvc_mlx._convert_bridge import randomize_bn_stats
from rvc_mlx.convert import convert_rmvpe_checkpoint, ensure_safetensors
from rvc_mlx.rmvpe import RMVPE


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
