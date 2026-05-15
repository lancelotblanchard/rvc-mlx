"""
Tests for the rmvpe module.

Each test class instantiates one MLX and one PyTorch MelSpectrogram with matched parameters, wraps each in a small
closure, and feeds the closure pair into the comparison framework. The PyTorch reference is a faithful copy of the
`MelSpectrogram` class from the RVC reference implementation.
"""

import librosa
import mlx.core as mx
import numpy as np
import pytest
import torch

from rvc_mlx.rmvpe import ConvBlockRes, MelSpectrogram, ResEncoderBlock

from .mlx_torch_comparison_framework import BaseOperationTest, OperationTestSuite
from .torch_bridge import (
    copy_conv_block_res,
    copy_res_encoder_block,
    randomize_bn_stats,
    set_eval,
    to_channels_first,
    to_channels_last,
)


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


# ----------------------------------------------------------------------------------------------------------------------
# RMVPE U-Net building blocks (ConvBlockRes, ResEncoderBlock).
# These tests use the paired-module pattern: build a PyTorch reference and an MLX implementation with matched
# hyperparameters, copy weights and BN running stats across, put both in eval mode, then compare forward outputs given
# the same input.
# ----------------------------------------------------------------------------------------------------------------------


class _TorchConvBlockRes(torch.nn.Module):
    """Faithful PyTorch reference for the RVC ConvBlockRes."""

    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels, momentum=momentum),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels, momentum=momentum),
            torch.nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        return self.conv(x) + self.shortcut(x)


class _TorchResEncoderBlock(torch.nn.Module):
    """Faithful PyTorch reference for the RVC ResEncoderBlock."""

    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = torch.nn.ModuleList()
        self.conv.append(_TorchConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(_TorchConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = torch.nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


def _build_conv_block_res_pair(in_channels, out_channels, momentum=0.01, seed=0):
    """Build a matched (mlx_fn, torch_fn) pair for ConvBlockRes, with eval mode and copied weights."""
    torch.manual_seed(seed)
    torch_mod = _TorchConvBlockRes(in_channels, out_channels, momentum=momentum)
    randomize_bn_stats(torch_mod, seed=seed)
    mlx_mod = ConvBlockRes(in_channels, out_channels, momentum=momentum)
    copy_conv_block_res(torch_mod, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    def mlx_fn(x):
        # x arrives as mx.array (channels-first) after framework conversion.
        out = mlx_mod(to_channels_last(x))
        return to_channels_first(out)

    def torch_fn(x):
        with torch.no_grad():
            return torch_mod(x)

    return mlx_fn, torch_fn


def _build_res_encoder_block_pair(
    in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01, seed=0
):
    """Build a matched (mlx_fn, torch_fn) pair for ResEncoderBlock returning only the pooled (or non-pooled) output.

    The pre-pool "skip" tensor is exercised by a separate pair (see `_build_res_encoder_block_skip_pair`) since
    `OperationTestSuite` compares a single output array.
    """
    torch.manual_seed(seed)
    torch_mod = _TorchResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
    randomize_bn_stats(torch_mod, seed=seed)
    mlx_mod = ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
    copy_res_encoder_block(torch_mod, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    def mlx_fn(x):
        out = mlx_mod(to_channels_last(x))
        if kernel_size is None:
            return to_channels_first(out)
        _, pooled = out
        return to_channels_first(pooled)

    def torch_fn(x):
        with torch.no_grad():
            out = torch_mod(x)
        if kernel_size is None:
            return out
        _, pooled = out
        return pooled

    return mlx_fn, torch_fn


def _build_res_encoder_block_skip_pair(
    in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01, seed=0
):
    """Same as `_build_res_encoder_block_pair` but returns the pre-pool skip tensor for comparison."""
    torch.manual_seed(seed)
    torch_mod = _TorchResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
    randomize_bn_stats(torch_mod, seed=seed)
    mlx_mod = ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
    copy_res_encoder_block(torch_mod, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    def mlx_fn(x):
        skip, _ = mlx_mod(to_channels_last(x))
        return to_channels_first(skip)

    def torch_fn(x):
        with torch.no_grad():
            skip, _ = torch_mod(x)
        return skip

    return mlx_fn, torch_fn


class TestRmvpeConvBlockResSameChannels(BaseOperationTest):
    """ConvBlockRes where in_channels == out_channels (identity shortcut path)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_conv_block_res_pair(in_channels=8, out_channels=8)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "conv_block_res_identity")

        rng = np.random.default_rng(0)
        cls.suite.add_test_case(
            name="identity_shortcut_basic",
            inputs={"x": rng.standard_normal((1, 8, 16, 32)).astype(np.float32)},
            description="Forward pass when shortcut is identity (no 1x1 conv)",
            atol=1e-4,
            rtol=1e-4,
        )
        cls.suite.add_test_case(
            name="identity_shortcut_batch",
            inputs={"x": rng.standard_normal((4, 8, 16, 32)).astype(np.float32)},
            description="Batched forward pass with identity shortcut",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeConvBlockResDifferentChannels(BaseOperationTest):
    """ConvBlockRes where in_channels != out_channels (1x1 conv shortcut path)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_conv_block_res_pair(in_channels=1, out_channels=16)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "conv_block_res_1x1_shortcut")

        rng = np.random.default_rng(1)
        cls.suite.add_test_case(
            name="shortcut_conv_basic",
            inputs={"x": rng.standard_normal((1, 1, 32, 128)).astype(np.float32)},
            description="Forward pass with 1x1 conv shortcut, RMVPE-shaped input",
            atol=1e-4,
            rtol=1e-4,
        )
        cls.suite.add_test_case(
            name="shortcut_conv_batch",
            inputs={"x": rng.standard_normal((2, 1, 32, 128)).astype(np.float32)},
            description="Batched forward pass with 1x1 conv shortcut",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeResEncoderBlockPooled(BaseOperationTest):
    """ResEncoderBlock with AvgPool2d kernel_size=(2, 2). Compares the pooled output."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_res_encoder_block_pair(
            in_channels=1, out_channels=16, kernel_size=(2, 2), n_blocks=1
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "res_encoder_block_pooled")

        rng = np.random.default_rng(2)
        cls.suite.add_test_case(
            name="pooled_single_block",
            inputs={"x": rng.standard_normal((1, 1, 32, 128)).astype(np.float32)},
            description="Single ConvBlockRes + AvgPool2d(2,2); pooled output",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeResEncoderBlockSkip(BaseOperationTest):
    """ResEncoderBlock with kernel_size=(2, 2). Compares the pre-pool skip output passed to the decoder."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_res_encoder_block_skip_pair(
            in_channels=1, out_channels=16, kernel_size=(2, 2), n_blocks=2
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "res_encoder_block_skip")

        rng = np.random.default_rng(3)
        cls.suite.add_test_case(
            name="skip_two_blocks",
            inputs={"x": rng.standard_normal((1, 1, 32, 128)).astype(np.float32)},
            description="Two ConvBlockRes then pool; we test the pre-pool skip tensor",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeResEncoderBlockNoPool(BaseOperationTest):
    """ResEncoderBlock with kernel_size=None (used inside `Intermediate`); compares the single output."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_res_encoder_block_pair(
            in_channels=16, out_channels=32, kernel_size=None, n_blocks=2
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "res_encoder_block_no_pool")

        rng = np.random.default_rng(4)
        cls.suite.add_test_case(
            name="no_pool_two_blocks",
            inputs={"x": rng.standard_normal((1, 16, 8, 32)).astype(np.float32)},
            description="Two ConvBlockRes without pooling (Intermediate-style)",
            atol=1e-4,
            rtol=1e-4,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
