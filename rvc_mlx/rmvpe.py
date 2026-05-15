from typing import Optional

import librosa
import mlx.core as mx
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
