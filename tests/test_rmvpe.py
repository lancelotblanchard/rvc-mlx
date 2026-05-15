"""
Tests for the rmvpe module.

Each test class instantiates one MLX and one PyTorch MelSpectrogram with matched parameters, wraps each in a small
closure, and feeds the closure pair into the comparison framework. The PyTorch reference is a faithful copy of the
`MelSpectrogram` class from the RVC reference implementation.
"""

import librosa
import numpy as np
import pytest
import torch

from rvc_mlx.rmvpe import MelSpectrogram

from .mlx_torch_comparison_framework import BaseOperationTest, OperationTestSuite


class _TorchMelSpectrogram(torch.nn.Module):
    """Faithful PyTorch reference matching the RVC implementation, used only as a comparison target in tests."""

    def __init__(
        self,
        is_half,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window: dict = {}
        mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        ).astype(np.float32)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis))
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new)
        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(-2)
            if resize < size:
                magnitude = torch.nn.functional.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[..., :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


def _build_pair(**init_params):
    """Construct an (mlx_fn, torch_fn) pair sharing the same init params."""
    mlx_mel = MelSpectrogram(**init_params)
    torch_mel = _TorchMelSpectrogram(**init_params)
    torch_mel.eval()

    def mlx_fn(audio, keyshift=0, speed=1, center=True):
        return mlx_mel(audio, keyshift=keyshift, speed=speed, center=center)

    def torch_fn(audio, keyshift=0, speed=1, center=True):
        with torch.no_grad():
            return torch_mel(audio, keyshift=keyshift, speed=speed, center=center)

    return mlx_fn, torch_fn


# RVC uses these MelSpectrogram parameters for the rmvpe pitch extractor.
RVC_DEFAULTS = dict(
    is_half=False,
    n_mel_channels=128,
    sampling_rate=16000,
    win_length=1024,
    hop_length=160,
    mel_fmin=30,
    mel_fmax=8000,
)


class TestRmvpeMelSpectrogramRvcDefaults(BaseOperationTest):
    """The exact configuration used by RVC's RMVPE pitch extractor."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_pair(**RVC_DEFAULTS)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "mel_spectrogram_rvc")

        rng = np.random.default_rng(0)

        cls.suite.add_test_case(
            name="batched_audio_default",
            inputs={"audio": rng.standard_normal((1, 16000)).astype(np.float32)},
            description="One-second batch of audio at the RVC sampling rate",
            atol=1e-3,
            rtol=1e-3,
        )

        cls.suite.add_test_case(
            name="batched_audio_no_center",
            inputs={
                "audio": rng.standard_normal((2, 16000)).astype(np.float32),
                "center": False,
            },
            description="Two-second batch with center=False (no STFT input padding)",
            atol=1e-3,
            rtol=1e-3,
        )

        cls.suite.add_test_case(
            name="batched_audio_keyshift_positive",
            inputs={
                "audio": rng.standard_normal((1, 16000)).astype(np.float32),
                "keyshift": 4,
            },
            description="keyshift > 0 (n_fft_new > n_fft, triggers slice down to target_size)",
            atol=1e-3,
            rtol=1e-3,
        )

        cls.suite.add_test_case(
            name="batched_audio_keyshift_negative",
            inputs={
                "audio": rng.standard_normal((1, 16000)).astype(np.float32),
                "keyshift": -5,
            },
            description="keyshift < 0 (n_fft_new < n_fft, triggers zero-pad branch)",
            atol=1e-3,
            rtol=1e-3,
        )

        cls.suite.add_test_case(
            name="batched_audio_speed_2",
            inputs={
                "audio": rng.standard_normal((1, 16000)).astype(np.float32),
                "speed": 2,
            },
            description="speed=2 doubles hop_length, halving the number of frames",
            atol=1e-3,
            rtol=1e-3,
        )


class TestRmvpeMelSpectrogramSmallConfig(BaseOperationTest):
    """A small configuration that exercises a different n_mel/n_fft ratio."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_pair(
            is_half=False,
            n_mel_channels=40,
            sampling_rate=22050,
            win_length=512,
            hop_length=128,
            mel_fmin=0,
            mel_fmax=None,
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "mel_spectrogram_small")

        rng = np.random.default_rng(1)

        cls.suite.add_test_case(
            name="single_audio",
            inputs={"audio": rng.standard_normal((1, 8000)).astype(np.float32)},
            description="Small mel config with default keyshift/speed",
            atol=1e-3,
            rtol=1e-3,
        )

        cls.suite.add_test_case(
            name="custom_n_fft",
            inputs={"audio": rng.standard_normal((1, 8000)).astype(np.float32)},
            description="Small config exercising matmul against a different mel basis shape",
            atol=1e-3,
            rtol=1e-3,
        )


class TestRmvpeMelSpectrogramExplicitNFft(BaseOperationTest):
    """A configuration with an explicit n_fft larger than win_length (windowing inside STFT)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_pair(
            is_half=False,
            n_mel_channels=80,
            sampling_rate=16000,
            win_length=400,
            hop_length=160,
            n_fft=512,
            mel_fmin=0,
            mel_fmax=8000,
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "mel_spectrogram_explicit_n_fft")

        rng = np.random.default_rng(2)

        cls.suite.add_test_case(
            name="audio_with_explicit_n_fft",
            inputs={"audio": rng.standard_normal((1, 8000)).astype(np.float32)},
            description="win_length < n_fft, STFT zero-pads window symmetrically",
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
