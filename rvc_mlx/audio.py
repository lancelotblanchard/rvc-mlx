"""
Audio I/O helpers for RVC inference.

Thin wrappers around `librosa.load` and `soundfile.write` with the conventions RVC expects:
  * Input is mono float32 at 16 kHz (anything else is resampled / downmixed).
  * Output is mono float32 written as 16-bit PCM WAV (the standard inference format).

Kept separate from `pipeline.py` so the inference path stays import-light: a caller that just wants `Pipeline.vc`
on pre-loaded MLX arrays doesn't pay for `librosa` / `soundfile` startup.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def load_audio_16k(path: str) -> np.ndarray:
    """
    Load a mono audio file resampled to 16 kHz as float32 in `[-1, 1]`.

    :param path: path to any libsndfile-readable file (WAV, FLAC, OGG, ...).
    :returns: 1D `np.ndarray` of `float32` samples.
    """
    import librosa

    audio, _sr = librosa.load(path, sr=16_000, mono=True)
    return audio.astype(np.float32)


def load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Load a mono audio file resampled to `target_sr`.

    :param path: path to any libsndfile-readable file.
    :param target_sr: target sample rate in Hz.
    :returns: `(audio, target_sr)` with `audio` as `float32` mono.
    """
    import librosa

    audio, _sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32), target_sr


def save_audio(path: str, audio: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """
    Write a 1D float audio array to disk.

    :param path: output file path. The container is inferred from the extension (.wav, .flac, ...).
    :param audio: 1D float audio in `[-1, 1]` (will be clipped before write).
    :param sr: sample rate in Hz.
    :param subtype: libsndfile subtype. Default `PCM_16` produces a 16-bit WAV — the format RVC's reference uses.
    """
    import soundfile as sf

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(path, audio, sr, subtype=subtype)
