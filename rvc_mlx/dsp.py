"""
Small NumPy signal helpers used by the conversion pipeline. Each mirrors exactly what the RVC WebUI does with
scipy / librosa / torch so the behaviour is pinned down in one place (and portable to the Swift / C++ runtimes).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# RVC removes DC / rumble before conversion: `scipy.signal.butter(N=5, Wn=48, btype="high", fs=16000)`.
HIGHPASS_B = np.array(
    [0.9699606451838447, -4.849803225919223, 9.699606451838447, -9.699606451838447, 4.849803225919223, -0.9699606451838447]
)
HIGHPASS_A = np.array(
    [1.0, -4.939001819168364, 9.757863526739543, -9.639544849413458, 4.761506797356209, -0.9408236532054606]
)


def highpass(audio: np.ndarray) -> np.ndarray:
    """Zero-phase 48 Hz high-pass (`scipy.signal.filtfilt` with RVC's Butterworth coefficients)."""
    return signal.filtfilt(HIGHPASS_B, HIGHPASS_A, audio)


def frame_rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """`librosa.feature.rms(y=y, frame_length=..., hop_length=...)[0]` (centered, zero-padded frames)."""
    pad = frame_length // 2
    y = np.pad(np.asarray(y, dtype=np.float64), (pad, pad), mode="constant")
    n_frames = 1 + (len(y) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n_frames, frame_length), strides=(y.strides[0] * hop_length, y.strides[0])
    )
    return np.sqrt(np.mean(frames**2, axis=-1))


def interp_linear(x: np.ndarray, size: int) -> np.ndarray:
    """`torch.nn.functional.interpolate(x[None, None], size=size, mode="linear")` (align_corners=False), 1-D."""
    n = len(x)
    pos = (np.arange(size) + 0.5) * (n / size) - 0.5
    pos = np.clip(pos, 0, n - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    return x[lo] * (1 - frac) + x[hi] * frac


def change_rms(source: np.ndarray, source_sr: int, target: np.ndarray, target_sr: int, rate: float) -> np.ndarray:
    """
    RVC's volume-envelope mix: scale `target` so its RMS envelope moves towards `source`'s. `rate=1` keeps the
    converted envelope untouched, `rate=0` imposes the input's envelope completely.
    """
    rms1 = frame_rms(source, source_sr // 2 * 2, source_sr // 2)
    rms2 = frame_rms(target, target_sr // 2 * 2, target_sr // 2)
    rms1 = interp_linear(rms1, target.shape[0])
    rms2 = np.maximum(interp_linear(rms2, target.shape[0]), 1e-6)
    return target * (np.power(rms1, 1 - rate) * np.power(rms2, rate - 1))
