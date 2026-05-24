from typing import Optional, List, Tuple, Union

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
    run on MLX arrays. The STFT is configured with `pad_mode="reflect"` and `onesided=True` to match
    `torch.stft`'s defaults for real input as used by RVC.
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
            pad_mode="reflect",
            onesided=True,
        )
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
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

    def __call__(self, x: mx.array) -> Union[Tuple[mx.array, mx.array], mx.array]:
        for block in self.conv:
            x = block(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: Optional[tuple],
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm(in_channels, momentum=momentum)
        self.layers = []
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    n_blocks=n_blocks,
                    momentum=momentum,
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        concat_tensors: List[mx.array] = []
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_inters: int,
        n_block: int,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_inters = n_inters
        self.layers = [ResEncoderBlock(in_channels, out_channels, None, n_block, momentum)]
        for i in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_block, momentum))

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int]],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = [ConvBlockRes(out_channels * 2, out_channels, momentum)]
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def __call__(self, x: mx.array, concat_tensor: mx.array) -> mx.array:
        x = self.conv1(x)
        # MLX uses channels-last (B, H, W, C), so concatenate along the last axis (channels). The PyTorch reference
        # uses `dim=1`, which is the channel dim in PyTorch's channels-first convention.
        x = mx.concatenate((x, concat_tensor), axis=-1)
        for i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_decoders: int,
        stride: Union[int, Tuple[int, int]],
        n_blocks: int,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.layers = []
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def __call__(self, x: mx.array, concat_tensors: List[mx.array]) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size: tuple,
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def __call__(self, x: mx.array) -> mx.array:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x
