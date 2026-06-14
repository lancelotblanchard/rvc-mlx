"""
Tests for the `rvc_mlx.infer` CLI entry point.

The CLI's job is to wire arg parsing -> model loading -> Pipeline.pipeline -> audio I/O. We don't have real
checkpoints in CI, so we monkeypatch each loader to return a stub, monkeypatch the audio I/O, and assert the
pipeline is invoked with the right shape of arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

from rvc_mlx import infer as infer_mod


class _StubHubert:
    def extract_features(self, audio, padding_mask=None, output_layer=None):  # noqa: ARG002
        import mlx.core as mx

        B, T = audio.shape
        return mx.zeros((B, T // 320, 768))


class _StubRMVPE:
    def __init__(self):
        self._f0 = np.zeros(80, dtype=np.float64)

    def infer_from_audio(self, audio, thred=0.03):  # noqa: ARG002
        return self._f0.copy()


class _StubSynthesizer:
    """Quacks like SynthesizerTrnMs768NSFsid: has `sr`, `.eval()`, and `.infer(...)` returning audio."""

    def __init__(self, sr=16000):
        self.sr = sr

    def eval(self):
        pass

    def infer(
        self,
        phone,
        phone_lengths,
        pitch,
        nsff0,
        sid,
        noise_z=None,
        rand_ini=None,
        noise_raw=None,
    ):
        import mlx.core as mx

        B, T, _ = phone.shape
        # Pretend the synthesizer upsamples 8x (matches the test pipeline config in test_pipeline.py).
        return mx.zeros((B, T * 8, 1)), None, None


def test_argument_parser_defaults():
    """Spot-check that the default CLI args match the documented contract."""
    p = infer_mod._build_arg_parser()
    args = p.parse_args(
        ["--input", "i.wav", "--output", "o.wav", "--voice", "v.pth", "--hubert", "h.pt", "--rmvpe", "r.pt"]
    )
    assert args.speaker_id == 0
    assert args.pitch_shift == 0
    assert args.rms_mix_rate == 0.25
    assert args.hubert_layer == 12
    assert args.verbose is False


def test_run_inference_wires_pipeline(monkeypatch, tmp_path):
    """End-to-end CLI smoke test with all heavy lifting stubbed out."""
    calls = {}

    def fake_hubert_from_pretrained(path):
        calls["hubert"] = path
        return _StubHubert()

    def fake_rmvpe_from_pretrained(path, is_half=False):  # noqa: ARG001
        calls["rmvpe"] = path
        return _StubRMVPE()

    def fake_synth_from_pretrained(path):
        calls["voice"] = path
        return _StubSynthesizer(sr=16000)

    def fake_load_audio_16k(path):
        calls["input"] = path
        return np.zeros(16_000, dtype=np.float32)

    def fake_save_audio(path, audio, sr, subtype="PCM_16"):
        calls["output"] = path
        calls["output_sr"] = sr
        calls["output_len"] = audio.shape[0]

    monkeypatch.setattr(infer_mod.HubertModel, "from_pretrained", staticmethod(fake_hubert_from_pretrained))
    monkeypatch.setattr(infer_mod.RMVPE, "from_pretrained", staticmethod(fake_rmvpe_from_pretrained))
    monkeypatch.setattr(infer_mod.SynthesizerTrnMs768NSFsid, "from_pretrained", staticmethod(fake_synth_from_pretrained))
    monkeypatch.setattr(infer_mod, "load_audio_16k", fake_load_audio_16k)
    monkeypatch.setattr(infer_mod, "save_audio", fake_save_audio)

    out_path = str(tmp_path / "out.wav")
    result = infer_mod.run_inference(
        input_path="in.wav",
        output_path=out_path,
        voice="voice.pth",
        hubert="hubert.pt",
        rmvpe="rmvpe.pt",
        speaker_id=3,
        pitch_shift=2,
        rms_mix_rate=0.5,
    )
    assert result == out_path
    assert calls["hubert"] == "hubert.pt"
    assert calls["rmvpe"] == "rmvpe.pt"
    assert calls["voice"] == "voice.pth"
    assert calls["input"] == "in.wav"
    assert calls["output"] == out_path
    assert calls["output_sr"] == 16000
    assert calls["output_len"] > 0


def test_main_parses_argv_and_returns_zero(monkeypatch):
    """`main(argv)` returns 0 on success; we stub `run_inference` so it doesn't actually do work."""
    captured = {}

    def fake_run_inference(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(infer_mod, "run_inference", fake_run_inference)
    rc = infer_mod.main(
        [
            "--input",
            "i.wav",
            "--output",
            "o.wav",
            "--voice",
            "v.pth",
            "--hubert",
            "h.pt",
            "--rmvpe",
            "r.pt",
            "--speaker-id",
            "5",
            "--pitch-shift",
            "-3",
        ]
    )
    assert rc == 0
    assert captured["speaker_id"] == 5
    assert captured["pitch_shift"] == -3
    assert captured["rms_mix_rate"] == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
