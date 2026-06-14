"""
Tests for `rvc_mlx.audio` I/O helpers.

Round-trip: write a known signal, read it back at the same sample rate, verify shape / dtype / approximate values.
We use `soundfile` directly to write the fixture (so the test isn't circular vs. our `save_audio`).
"""

from __future__ import annotations

import numpy as np
import pytest

from rvc_mlx.audio import load_audio, load_audio_16k, save_audio


def _write_fixture_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write a fixture WAV via soundfile directly, sidestepping our `save_audio` so the test is non-circular."""
    import soundfile as sf

    sf.write(path, audio, sr, subtype="PCM_16")


class TestLoadAudio16k:
    def test_loads_at_16k(self, tmp_path):
        sr_src = 8_000
        audio = np.sin(2 * np.pi * 440 * np.arange(sr_src) / sr_src).astype(np.float32)
        wav = str(tmp_path / "sine.wav")
        _write_fixture_wav(wav, audio, sr_src)

        loaded = load_audio_16k(wav)

        # Resampled to 16 kHz -> twice as many samples (approximately).
        assert loaded.dtype == np.float32
        assert loaded.ndim == 1
        assert abs(loaded.shape[0] - 2 * sr_src) <= 16  # allow a few samples of resampler boundary slack

    def test_downmixes_stereo(self, tmp_path):
        sr = 16_000
        # Stereo: left = +sine, right = -sine. Mono mix should be near zero.
        t = np.arange(sr) / sr
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        stereo = np.stack([sine, -sine], axis=1)
        wav = str(tmp_path / "stereo.wav")
        _write_fixture_wav(wav, stereo, sr)

        loaded = load_audio_16k(wav)
        assert loaded.ndim == 1
        # The cancellation isn't perfect because the downmix happens before resampling, but the RMS should be near 0.
        assert np.sqrt(np.mean(loaded**2)) < 0.05


class TestLoadAudioGenericSr:
    def test_load_at_target_sr(self, tmp_path):
        audio = np.zeros(8000, dtype=np.float32)
        audio[::2] = 0.5
        wav = str(tmp_path / "fixture.wav")
        _write_fixture_wav(wav, audio, 8000)

        loaded, sr = load_audio(wav, target_sr=32_000)
        assert sr == 32_000
        # 4x upsample of 8000 samples -> 32000.
        assert abs(loaded.shape[0] - 32_000) <= 64


class TestSaveAudio:
    def test_round_trip(self, tmp_path):
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(16_000) * 0.1).astype(np.float32)
        wav = str(tmp_path / "rt.wav")
        save_audio(wav, audio, sr=16_000)

        import soundfile as sf

        loaded, sr_out = sf.read(wav, dtype="float32")
        assert sr_out == 16_000
        assert loaded.shape == audio.shape
        # PCM_16 has ~5e-5 quantization error; allow some slack.
        np.testing.assert_allclose(loaded, audio, atol=1e-3)

    def test_clips_out_of_range_values(self, tmp_path):
        """`save_audio` clips to [-1, 1] before writing; verify writing 2.0 yields 1.0 (after PCM16 round-trip)."""
        wav = str(tmp_path / "clipped.wav")
        save_audio(wav, np.array([2.0, -2.0, 0.0, 0.5], dtype=np.float32), sr=16_000)

        import soundfile as sf

        loaded, _ = sf.read(wav, dtype="float32")
        # PCM_16's max representable value is just under 1.0; clipping ensures values <= 1.
        assert loaded.max() <= 1.0 + 1e-3
        assert loaded.min() >= -1.0 - 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
