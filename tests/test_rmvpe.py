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

from rvc_mlx.rmvpe import (
    BiGRU,
    ConvBlockRes,
    Decoder,
    DeepUnet,
    E2E,
    Encoder,
    Intermediate,
    MelSpectrogram,
    ResDecoderBlock,
    ResEncoderBlock,
    RMVPE,
)

from .mlx_torch_comparison_framework import BaseOperationTest, OperationTestSuite
from .torch_bridge import (
    copy_bi_gru,
    copy_conv_block_res,
    copy_decoder,
    copy_deep_unet,
    copy_e2e,
    copy_encoder,
    copy_intermediate,
    copy_res_decoder_block,
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


# ----------------------------------------------------------------------------------------------------------------------
# Encoder, Intermediate, ResDecoderBlock, Decoder, DeepUnet.
# Same paired-module strategy as ConvBlockRes / ResEncoderBlock above: build matched torch+MLX modules, copy weights and
# BN running stats, run forward in eval mode, and compare. All inputs are channels-first PyTorch-shaped (B, C, H, W);
# the MLX wrapper transposes to channels-last on the way in and back on the way out.
# ----------------------------------------------------------------------------------------------------------------------


class _TorchEncoder(torch.nn.Module):
    """Faithful PyTorch reference for the RVC encoder (called `RmvpeEncoder` in the original)."""

    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = torch.nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = torch.nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(
                _TorchResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
            )
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class _TorchIntermediate(torch.nn.Module):
    """Faithful PyTorch reference for the RVC Intermediate block."""

    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = torch.nn.ModuleList()
        self.layers.append(_TorchResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(_TorchResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _TorchResDecoderBlock(torch.nn.Module):
    """Faithful PyTorch reference for the RVC ResDecoderBlock."""

    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels, momentum=momentum),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.ModuleList()
        self.conv2.append(_TorchConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(_TorchConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class _TorchDecoder(torch.nn.Module):
    """Faithful PyTorch reference for the RVC decoder (`RmvpeDecoder`)."""

    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                _TorchResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class _TorchDeepUnet(torch.nn.Module):
    """Faithful PyTorch reference for the RVC DeepUnet."""

    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.encoder = _TorchEncoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = _TorchIntermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = _TorchDecoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


# ---------- Builders ---------------------------------------------------------------------------------------------------


def _build_encoder_pair(in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, seed=0):
    torch.manual_seed(seed)
    torch_mod = _TorchEncoder(in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels)
    randomize_bn_stats(torch_mod, seed=seed)
    mlx_mod = Encoder(in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels)
    copy_encoder(torch_mod, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    # Encoder returns (x, concat_tensors); the framework compares a single array, so we test the bottleneck output and
    # each skip tensor with separate suites. This builder returns the bottleneck-output wrapper.
    def mlx_fn(x):
        out, _ = mlx_mod(to_channels_last(x))
        return to_channels_first(out)

    def torch_fn(x):
        with torch.no_grad():
            out, _ = torch_mod(x)
        return out

    return mlx_fn, torch_fn, torch_mod, mlx_mod


def _build_encoder_skip_pair(skip_index, *args, **kwargs):
    """Build a pair that returns concat_tensors[skip_index] for shape-level comparison."""
    torch.manual_seed(kwargs.get("seed", 0))
    torch_mod = _TorchEncoder(*args, **{k: v for k, v in kwargs.items() if k != "seed"})
    randomize_bn_stats(torch_mod, seed=kwargs.get("seed", 0))
    mlx_mod = Encoder(*args, **{k: v for k, v in kwargs.items() if k != "seed"})
    copy_encoder(torch_mod, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    def mlx_fn(x):
        _, concat = mlx_mod(to_channels_last(x))
        return to_channels_first(concat[skip_index])

    def torch_fn(x):
        with torch.no_grad():
            _, concat = torch_mod(x)
        return concat[skip_index]

    return mlx_fn, torch_fn


def _build_intermediate_pair(in_channels, out_channels, n_inters, n_blocks, seed=0):
    torch.manual_seed(seed)
    torch_mod = _TorchIntermediate(in_channels, out_channels, n_inters, n_blocks)
    randomize_bn_stats(torch_mod, seed=seed)
    mlx_mod = Intermediate(in_channels, out_channels, n_inters, n_blocks)
    copy_intermediate(torch_mod, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    def mlx_fn(x):
        return to_channels_first(mlx_mod(to_channels_last(x)))

    def torch_fn(x):
        with torch.no_grad():
            return torch_mod(x)

    return mlx_fn, torch_fn


def _build_res_decoder_block_pair(in_channels, out_channels, stride, n_blocks=1, seed=0):
    torch.manual_seed(seed)
    torch_mod = _TorchResDecoderBlock(in_channels, out_channels, stride, n_blocks)
    randomize_bn_stats(torch_mod, seed=seed)
    mlx_mod = ResDecoderBlock(in_channels, out_channels, stride, n_blocks)
    copy_res_decoder_block(torch_mod, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    def mlx_fn(x, concat_tensor):
        out = mlx_mod(to_channels_last(x), to_channels_last(concat_tensor))
        return to_channels_first(out)

    def torch_fn(x, concat_tensor):
        with torch.no_grad():
            return torch_mod(x, concat_tensor)

    return mlx_fn, torch_fn


def _build_decoder_pair(in_channels, n_decoders, stride, n_blocks, seed=0):
    """
    Build a Decoder pair. The framework calls the wrapper with kwargs, so the wrapper takes named skip tensors
    `skip0`..`skipN-1` (encoder order: skip0 is the largest, consumed last by the decoder). The wrapper assembles them
    into a list to match the real `Decoder.__call__` signature.
    """
    torch.manual_seed(seed)
    torch_mod = _TorchDecoder(in_channels, n_decoders, stride, n_blocks)
    randomize_bn_stats(torch_mod, seed=seed)
    mlx_mod = Decoder(in_channels, n_decoders, stride, n_blocks)
    copy_decoder(torch_mod, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    def mlx_fn(x, **skips):
        skip_list = [skips[f"skip{i}"] for i in range(n_decoders)]
        skips_cl = [to_channels_last(s) for s in skip_list]
        out = mlx_mod(to_channels_last(x), skips_cl)
        return to_channels_first(out)

    def torch_fn(x, **skips):
        skip_list = [skips[f"skip{i}"] for i in range(n_decoders)]
        with torch.no_grad():
            return torch_mod(x, skip_list)

    return mlx_fn, torch_fn


def _build_deep_unet_pair(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels=1, en_out_channels=16, seed=0):
    torch.manual_seed(seed)
    torch_mod = _TorchDeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
    randomize_bn_stats(torch_mod, seed=seed)
    mlx_mod = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
    copy_deep_unet(torch_mod, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    def mlx_fn(x):
        return to_channels_first(mlx_mod(to_channels_last(x)))

    def torch_fn(x):
        with torch.no_grad():
            return torch_mod(x)

    return mlx_fn, torch_fn


# ---------- Tests ------------------------------------------------------------------------------------------------------


class TestRmvpeEncoderBottleneck(BaseOperationTest):
    """Encoder output (the bottleneck tensor passed to Intermediate)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn, _, _ = _build_encoder_pair(
            in_channels=1, in_size=128, n_encoders=3, kernel_size=(2, 2), n_blocks=1, out_channels=8
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "encoder_bottleneck")

        rng = np.random.default_rng(10)
        cls.suite.add_test_case(
            name="bottleneck_basic",
            inputs={"x": rng.standard_normal((1, 1, 32, 128)).astype(np.float32)},
            description="3-layer Encoder; spatial dims divide by 2^3 = 8",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeEncoderSkip0(BaseOperationTest):
    """Encoder skip tensor at layer 0 (largest spatial size)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_encoder_skip_pair(
            0, 1, 128, 3, (2, 2), 1, 8
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "encoder_skip_0")

        rng = np.random.default_rng(11)
        cls.suite.add_test_case(
            name="skip_0",
            inputs={"x": rng.standard_normal((1, 1, 32, 128)).astype(np.float32)},
            description="First skip tensor (pre-pool output of layer 0)",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeEncoderSkipLast(BaseOperationTest):
    """Encoder skip tensor at the last layer (smallest spatial size)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_encoder_skip_pair(
            2, 1, 128, 3, (2, 2), 1, 8
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "encoder_skip_last")

        rng = np.random.default_rng(12)
        cls.suite.add_test_case(
            name="skip_last",
            inputs={"x": rng.standard_normal((1, 1, 32, 128)).astype(np.float32)},
            description="Last skip tensor (pre-pool output of the deepest encoder layer)",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeIntermediate(BaseOperationTest):
    """Intermediate block (stack of ResEncoderBlock with kernel_size=None)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_intermediate_pair(
            in_channels=16, out_channels=32, n_inters=2, n_blocks=1
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "intermediate")

        rng = np.random.default_rng(13)
        cls.suite.add_test_case(
            name="intermediate_basic",
            inputs={"x": rng.standard_normal((1, 16, 8, 16)).astype(np.float32)},
            description="Intermediate increases channels (16 -> 32) without changing spatial size",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeResDecoderBlock(BaseOperationTest):
    """ResDecoderBlock (transpose conv upsample + concat with skip + ConvBlockRes stack)."""

    @classmethod
    def setup_class(cls):
        # Stride (2, 2) doubles each spatial dim. Input (B, in, 8, 8) and skip (B, out, 16, 16).
        mlx_fn, torch_fn = _build_res_decoder_block_pair(
            in_channels=32, out_channels=16, stride=(2, 2), n_blocks=1
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "res_decoder_block")

        rng = np.random.default_rng(14)
        cls.suite.add_test_case(
            name="res_decoder_basic",
            inputs={
                "x": rng.standard_normal((1, 32, 8, 8)).astype(np.float32),
                "concat_tensor": rng.standard_normal((1, 16, 16, 16)).astype(np.float32),
            },
            description="Upsample 8x8 -> 16x16, concat with 16x16 skip, run ConvBlockRes",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeDecoder(BaseOperationTest):
    """Full Decoder: a list of ResDecoderBlocks consuming the encoder's skip tensors in reverse order."""

    @classmethod
    def setup_class(cls):
        # Build a 3-layer decoder mirroring a 3-layer encoder. In_channels halve each step: 32 -> 16 -> 8 -> 4.
        # Skip tensors expected per layer (channels-first), in encoder order (decoder consumes them in reverse).
        # After encoder layers 0/1/2 starting from (32, 128): pooled produces (8, 64), (16, 32), (32, 16).
        # We mimic that here with synthetic skips.
        mlx_fn, torch_fn = _build_decoder_pair(
            in_channels=32,
            n_decoders=3,
            stride=(2, 2),
            n_blocks=1,
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "decoder")

        # Decoder iteration i: ConvTranspose maps in_channels // 2^i -> in_channels // 2^(i+1), then concatenates with
        # concat_tensors[-1 - i] (encoder skip from layer N-1-i). The skip tensor's channel count must match the
        # upsampled output's channel count, and its spatial dims must match the upsampled spatial dims (input * 2^(i+1)).
        rng = np.random.default_rng(15)
        cls.suite.add_test_case(
            name="decoder_basic",
            inputs={
                "x": rng.standard_normal((1, 32, 4, 4)).astype(np.float32),
                # skip0 is consumed at decoder step 2 (deepest unroll): spatial (32, 32), 4 channels.
                "skip0": rng.standard_normal((1, 4, 32, 32)).astype(np.float32),
                # skip1 is consumed at decoder step 1: spatial (16, 16), 8 channels.
                "skip1": rng.standard_normal((1, 8, 16, 16)).astype(np.float32),
                # skip2 is consumed at decoder step 0 (first): spatial (8, 8), 16 channels.
                "skip2": rng.standard_normal((1, 16, 8, 8)).astype(np.float32),
            },
            description="3-step decoder unrolling 4x4 -> 32x32, concatenating with three skip tensors",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeDeepUnet(BaseOperationTest):
    """End-to-end DeepUnet: Encoder + Intermediate + Decoder."""

    @classmethod
    def setup_class(cls):
        # Match RVC default architecture but with a smaller depth/channel count to keep the test cheap.
        mlx_fn, torch_fn = _build_deep_unet_pair(
            kernel_size=(2, 2),
            n_blocks=1,
            en_de_layers=3,
            inter_layers=2,
            in_channels=1,
            en_out_channels=8,
        )
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "deep_unet")

        rng = np.random.default_rng(16)
        cls.suite.add_test_case(
            name="deep_unet_basic",
            inputs={"x": rng.standard_normal((1, 1, 32, 128)).astype(np.float32)},
            description="Full DeepUnet forward; output has the same spatial size as input and en_out_channels channels",
            atol=1e-4,
            rtol=1e-4,
        )


# ----------------------------------------------------------------------------------------------------------------------
# BiGRU and E2E
# ----------------------------------------------------------------------------------------------------------------------


class _TorchBiGRU(torch.nn.Module):
    """Faithful PyTorch reference: a bidirectional, multi-layer GRU."""

    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class _TorchE2E(torch.nn.Module):
    """Faithful PyTorch reference for the RVC E2E pitch network (n_gru > 0 path only)."""

    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.unet = _TorchDeepUnet(
            kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels
        )
        self.cnn = torch.nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = torch.nn.Sequential(
            _TorchBiGRU(3 * 128, 256, n_gru),
            torch.nn.Linear(512, 360),
            torch.nn.Dropout(0.25),
            torch.nn.Sigmoid(),
        )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


def _build_bi_gru_pair(input_features, hidden_features, num_layers, seed=0):
    torch.manual_seed(seed)
    torch_mod = _TorchBiGRU(input_features, hidden_features, num_layers)
    mlx_mod = BiGRU(input_features, hidden_features, num_layers)
    copy_bi_gru(torch_mod.gru, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    def mlx_fn(x):
        return mlx_mod(x)

    def torch_fn(x):
        with torch.no_grad():
            return torch_mod(x)

    return mlx_fn, torch_fn


def _build_e2e_pair(
    n_blocks, n_gru, kernel_size, en_de_layers, inter_layers, in_channels=1, en_out_channels=16, seed=0
):
    torch.manual_seed(seed)
    torch_mod = _TorchE2E(
        n_blocks, n_gru, kernel_size, en_de_layers, inter_layers, in_channels, en_out_channels
    )
    randomize_bn_stats(torch_mod, seed=seed)
    mlx_mod = E2E(
        n_blocks, n_gru, kernel_size, en_de_layers, inter_layers, in_channels, en_out_channels
    )
    copy_e2e(torch_mod, mlx_mod)
    set_eval(torch_mod, mlx_mod)

    def mlx_fn(mel):
        return mlx_mod(mel)

    def torch_fn(mel):
        with torch.no_grad():
            return torch_mod(mel)

    return mlx_fn, torch_fn, mlx_mod, torch_mod


class TestRmvpeBiGRUSingleLayer(BaseOperationTest):
    """One-layer bidirectional GRU."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_bi_gru_pair(input_features=16, hidden_features=8, num_layers=1)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "bi_gru_1_layer")

        rng = np.random.default_rng(20)
        cls.suite.add_test_case(
            name="bi_gru_basic",
            inputs={"x": rng.standard_normal((1, 12, 16)).astype(np.float32)},
            description="Single-layer bidirectional GRU; output is (B, T, 2*hidden)",
            atol=1e-4,
            rtol=1e-4,
        )
        cls.suite.add_test_case(
            name="bi_gru_batched",
            inputs={"x": rng.standard_normal((3, 20, 16)).astype(np.float32)},
            description="Batched single-layer bidirectional GRU",
            atol=1e-4,
            rtol=1e-4,
        )


class TestRmvpeBiGRUMultiLayer(BaseOperationTest):
    """Two-layer bidirectional GRU (the RVC RMVPE configuration uses n_gru=1, but multi-layer should also work)."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn = _build_bi_gru_pair(input_features=16, hidden_features=8, num_layers=2)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "bi_gru_2_layers")

        rng = np.random.default_rng(21)
        cls.suite.add_test_case(
            name="bi_gru_two_layers",
            inputs={"x": rng.standard_normal((1, 12, 16)).astype(np.float32)},
            description="Two-layer bidirectional GRU; second layer takes 2*hidden as input",
            atol=1e-4,
            rtol=1e-4,
        )


# The RVC RMVPE E2E uses en_de_layers=5, inter_layers=4, en_out_channels=16, n_blocks=4, n_gru=1. That's expensive,
# so we exercise a shrunk version that still hits the full code path (DeepUnet + Conv + BiGRU + Linear + sigmoid).
_E2E_TEST_CONFIG = dict(
    n_blocks=1,
    n_gru=1,
    kernel_size=(2, 2),
    en_de_layers=3,
    inter_layers=2,
    in_channels=1,
    en_out_channels=8,
)


class TestRmvpeE2E(BaseOperationTest):
    """End-to-end E2E pitch network forward pass."""

    @classmethod
    def setup_class(cls):
        mlx_fn, torch_fn, _, _ = _build_e2e_pair(**_E2E_TEST_CONFIG)
        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "e2e")

        rng = np.random.default_rng(22)
        # Time dim must be a multiple of 2 ** en_de_layers = 8 for the U-Net to downsample evenly. n_mel is fixed at 128
        # by the reference (the in_size of the U-Net).
        cls.suite.add_test_case(
            name="e2e_basic",
            inputs={"mel": rng.standard_normal((1, 128, 32)).astype(np.float32)},
            description="E2E forward; output shape (B, T, 360)",
            atol=1e-3,
            rtol=1e-3,
        )


# ----------------------------------------------------------------------------------------------------------------------
# RMVPE orchestration (mel2hidden, to_local_average_cents, decode, infer_from_audio).
# ----------------------------------------------------------------------------------------------------------------------


class _TorchRMVPE:
    """
    PyTorch reference orchestration class mirroring the RVC `RMVPE`. We reproduce only the inference path
    (mel extraction, padding to multiple of 32, model forward, salience decoding) and not the checkpoint-loading or
    JIT-export plumbing, since those are unrelated to the numerical behavior under test.
    """

    _CENTS_OFFSET = 1997.3794084376191
    _CENTS_STEP = 20
    _NUM_BINS = 360
    _PAD = 4

    def __init__(self, model, is_half=False):
        self.is_half = is_half
        self.model = model
        self.mel_extractor = _TorchMelSpectrogram(
            is_half=is_half,
            n_mel_channels=128,
            sampling_rate=16000,
            win_length=1024,
            hop_length=160,
            mel_fmin=30,
            mel_fmax=8000,
        )
        self.mel_extractor.eval()
        cents_mapping = self._CENTS_STEP * np.arange(self._NUM_BINS) + self._CENTS_OFFSET
        self.cents_mapping = np.pad(cents_mapping, (self._PAD, self._PAD))

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = torch.nn.functional.pad(mel, (0, n_pad), mode="constant")
            if self.is_half:
                mel = mel.half()
            hidden = self.model(mel)
            return hidden[..., :n_frames, :]

    def to_local_average_cents(self, salience, thred=0.05):
        salience_np = salience.detach().cpu().numpy() if isinstance(salience, torch.Tensor) else np.asarray(salience)
        center = np.argmax(salience_np, axis=-1)
        padded = np.pad(salience_np, [(0, 0)] * (salience_np.ndim - 1) + [(self._PAD, self._PAD)])
        window_size = 2 * self._PAD + 1
        offsets = np.arange(window_size)
        idx = center[..., None] + offsets
        gathered = np.take_along_axis(padded, idx, axis=-1)
        cents_window = self.cents_mapping[idx]
        product_sum = np.sum(gathered * cents_window, axis=-1)
        weight_sum = np.sum(gathered, axis=-1)
        weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
        devided = product_sum / weight_sum
        maxx = np.max(salience_np, axis=-1)
        devided = np.where(maxx <= thred, 0.0, devided)
        return devided

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0 = np.where(f0 == 10, 0.0, f0)
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        mel = self.mel_extractor(audio)
        hidden = self.mel2hidden(mel)
        return self.decode(hidden, thred=thred)


def _build_rmvpe_pair(**e2e_config):
    """Construct matched MLX and PyTorch RMVPE orchestrators with weight-bridged E2E networks."""
    seed = e2e_config.pop("seed", 0)
    torch.manual_seed(seed)
    torch_e2e = _TorchE2E(**e2e_config)
    randomize_bn_stats(torch_e2e, seed=seed)
    mlx_e2e = E2E(**e2e_config)
    copy_e2e(torch_e2e, mlx_e2e)
    # Both networks must be in eval mode; otherwise BatchNorm uses batch statistics (training mode) and the bridge's
    # running-stat copy is ignored. This was a subtle source of ~0.28 max-abs divergence before being added.
    set_eval(torch_e2e, mlx_e2e)

    mlx_rmvpe = RMVPE(model=mlx_e2e, is_half=False)
    torch_rmvpe = _TorchRMVPE(model=torch_e2e, is_half=False)
    return mlx_rmvpe, torch_rmvpe


class TestRmvpeMel2Hidden(BaseOperationTest):
    """`mel2hidden` pads time to a multiple of 32, runs the network, then trims back to the original frame count."""

    @classmethod
    def setup_class(cls):
        mlx_rmvpe, torch_rmvpe = _build_rmvpe_pair(**_E2E_TEST_CONFIG)

        def mlx_fn(mel):
            return mlx_rmvpe.mel2hidden(mel)

        def torch_fn(mel):
            return torch_rmvpe.mel2hidden(mel)

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "mel2hidden")

        rng = np.random.default_rng(30)
        # Pick a non-multiple of 32 so the padding branch fires.
        cls.suite.add_test_case(
            name="mel2hidden_pads",
            inputs={"mel": rng.standard_normal((1, 128, 50)).astype(np.float32)},
            description="50 frames -> padded to 64 -> trimmed back to 50",
            atol=1e-3,
            rtol=1e-3,
        )
        cls.suite.add_test_case(
            name="mel2hidden_already_aligned",
            inputs={"mel": rng.standard_normal((1, 128, 32)).astype(np.float32)},
            description="No-pad path: 32 frames is already a multiple of 32",
            atol=1e-3,
            rtol=1e-3,
        )


class TestRmvpeToLocalAverageCents(BaseOperationTest):
    """`to_local_average_cents` decoding is pure NumPy and must match exactly."""

    @classmethod
    def setup_class(cls):
        mlx_rmvpe, torch_rmvpe = _build_rmvpe_pair(**_E2E_TEST_CONFIG)

        def mlx_fn(salience, thred):
            return mlx_rmvpe.to_local_average_cents(salience, thred=thred)

        def torch_fn(salience, thred):
            return torch_rmvpe.to_local_average_cents(salience, thred=thred)

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "to_local_average_cents")

        rng = np.random.default_rng(31)
        salience = rng.uniform(size=(1, 16, 360)).astype(np.float32)
        # Some frames intentionally low-salience so the threshold branch fires.
        salience[0, 4] *= 0.001
        salience[0, 9] *= 0.001
        cls.suite.add_test_case(
            name="local_average_basic",
            inputs={"salience": salience, "thred": 0.05},
            description="Decoded cents per frame, with some frames below the salience threshold",
            atol=1e-6,
            rtol=1e-6,
        )


class TestRmvpeDecode(BaseOperationTest):
    """`decode` composes `to_local_average_cents` with the cents -> Hz formula."""

    @classmethod
    def setup_class(cls):
        mlx_rmvpe, torch_rmvpe = _build_rmvpe_pair(**_E2E_TEST_CONFIG)

        def mlx_fn(hidden, thred):
            return mlx_rmvpe.decode(hidden, thred=thred)

        def torch_fn(hidden, thred):
            return torch_rmvpe.decode(hidden, thred=thred)

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "decode")

        rng = np.random.default_rng(32)
        cls.suite.add_test_case(
            name="decode_basic",
            inputs={
                "hidden": rng.uniform(size=(1, 16, 360)).astype(np.float32),
                "thred": 0.03,
            },
            description="Decode random salience to f0 (Hz). Unvoiced frames map to 0.",
            atol=1e-5,
            rtol=1e-5,
        )


class TestRmvpeInferFromAudio(BaseOperationTest):
    """Full RMVPE pipeline: raw audio -> mel -> hidden -> f0."""

    @classmethod
    def setup_class(cls):
        mlx_rmvpe, torch_rmvpe = _build_rmvpe_pair(**_E2E_TEST_CONFIG)

        def mlx_fn(audio, thred):
            return mlx_rmvpe.infer_from_audio(audio, thred=thred)

        def torch_fn(audio, thred):
            return torch_rmvpe.infer_from_audio(audio, thred=thred)

        cls.suite = OperationTestSuite(mlx_fn, torch_fn, "infer_from_audio")

        rng = np.random.default_rng(33)
        # 1 s of 16 kHz audio. With hop_length=160 we get 101 mel frames (with center=True).
        cls.suite.add_test_case(
            name="infer_basic",
            inputs={
                "audio": rng.standard_normal((1, 16000)).astype(np.float32),
                "thred": 0.03,
            },
            description="End-to-end inference; random weights so output values are arbitrary, but MLX and PyTorch agree",
            atol=1e-2,
            rtol=1e-2,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
