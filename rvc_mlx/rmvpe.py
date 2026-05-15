from typing import Optional

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from rvc_mlx.stft import stft
from rvc_mlx.utils import pad_constant
from rvc_mlx.windows import hann


class MelSpectrogram:
    """
    Mel spectrogram extractor matching the RVC reference implementation. Computes the log-mel spectrogram of an audio
    signal, with optional pitch-shifting (`keyshift`) and time-stretching (`speed`).

    The mel filterbank is precomputed using `librosa.filters.mel(..., htk=True)` to remain bit-exact with the original
    RVC reference implementation, which uses the same call. Subsequent operations (STFT, magnitude, mel projection, log)
    run on MLX arrays.

    Note: `torch.stft(..., return_complex=True)` returns a one-sided spectrum by default for real input. The MLX `stft`
    implementation in `rvc_mlx.stft` only supports `onesided=False`, so the full spectrum is sliced down to
    `n_fft // 2 + 1` frequency bins here before computing the magnitude.
    """

    def __init__(
        self,
        is_half: bool,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: Optional[int] = None,
        mel_fmin: float = 0,
        mel_fmax: Optional[float] = None,
        clamp: float = 1e-5,
    ):
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        ).astype(np.float32)
        self.mel_basis = mx.array(mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half
        self.hann_window: dict = {}

    def __call__(
        self,
        audio: mx.array,
        keyshift: int = 0,
        speed: int = 1,
        center: bool = True,
    ) -> mx.array:
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        keyshift_key = str(keyshift)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = hann(win_length_new, sym=False)

        fft = stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
        )
        # Match torch.stft's default onesided=True for real input by slicing the full two-sided spectrum down to
        # n_fft // 2 + 1 frequency bins along the frequency axis (second-to-last).
        size_new = n_fft_new // 2 + 1
        fft = fft[..., :size_new, :]
        magnitude = mx.sqrt(fft.real**2 + fft.imag**2)

        if keyshift != 0:
            target_size = self.n_fft // 2 + 1
            resize = magnitude.shape[-2]
            if resize < target_size:
                # Pad the frequency axis (second-to-last) on the right with zeros so it matches `target_size`.
                magnitude = pad_constant(magnitude, (0, 0, 0, target_size - resize), value=0.0)
            magnitude = magnitude[..., :target_size, :] * self.win_length / win_length_new

        mel_output = mx.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.astype(mx.float16)
        log_mel_spec = mx.log(mx.clip(mel_output, a_min=self.clamp, a_max=None))
        return log_mel_spec


class ConvBlockRes(nn.Module):
    """
    Residual convolutional block used throughout RVC's RMVPE U-Net. Two 3x3 conv + BN + ReLU stages followed by an
    additive shortcut. If `in_channels != out_channels`, the shortcut is a 1x1 conv that matches dimensions; otherwise
    it is the identity.

    Note: MLX uses channels-last convention for 2D convolutions. Callers should provide input in shape
    `(B, H, W, C)` rather than PyTorch's `(B, C, H, W)`. The internal structure mirrors the PyTorch reference so weights
    can be copied with a simple per-attribute transposition (see tests).
    """

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self._has_shortcut = in_channels != out_channels
        if self._has_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = self.shortcut(x) if self._has_shortcut else x
        return self.conv(x) + residual


class ResEncoderBlock(nn.Module):
    """
    A stack of `n_blocks` `ConvBlockRes` modules followed by an optional `AvgPool2d`. When `kernel_size` is `None`, no
    pooling is applied and the block returns a single tensor (used by `Intermediate`). Otherwise it returns a
    `(skip, pooled)` tuple, where `skip` is the pre-pool feature map kept for the decoder and `pooled` is the
    downsampled output that continues through the encoder.

    Input/output are channels-last `(B, H, W, C)` in MLX convention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[tuple],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = [ConvBlockRes(in_channels, out_channels, momentum)]
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def __call__(self, x: mx.array):
        for block in self.conv:
            x = block(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x
