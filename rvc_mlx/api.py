"""
High-level API.

    from rvc_mlx import RVC, Voice

    rvc = RVC.from_pretrained("models/")                  # hubert.safetensors + rmvpe.safetensors
    voice = Voice.load("models/voices/alice.safetensors")
    audio, sr = rvc.convert_file("in.wav", "out.wav", voice, pitch=0)
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from rvc_mlx.hubert import HubertModel
from rvc_mlx.pipeline import Pipeline, PipelineConfig
from rvc_mlx.rmvpe import RMVPE
from rvc_mlx.voice import Voice

_HUBERT_NAMES = ("hubert.safetensors", "hubert_base.safetensors", "contentvec.safetensors")
_RMVPE_NAMES = ("rmvpe.safetensors",)


def _find(models_dir: str, names) -> str:
    for n in names:
        p = os.path.join(models_dir, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {names} found in {models_dir!r}. Convert them first: `rvc-mlx convert-base`.")


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Read any file soundfile/librosa understand, returned as mono float32 at its native rate."""
    import librosa

    audio, sr = librosa.load(path, sr=None, mono=True)
    return audio.astype(np.float32), int(sr)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq").astype(np.float32)


class RVC:
    """The shared, voice-independent half of RVC (content encoder + pitch estimator) plus conversion entry points."""

    def __init__(self, hubert: HubertModel, rmvpe: RMVPE, config: Optional[PipelineConfig] = None):
        self.hubert = hubert
        self.rmvpe = rmvpe
        self.config = config or PipelineConfig()

    @classmethod
    def from_pretrained(
        cls,
        models_dir: Optional[str] = None,
        hubert_path: Optional[str] = None,
        rmvpe_path: Optional[str] = None,
        dtype: Union[str, mx.Dtype] = mx.float32,
        config: Optional[PipelineConfig] = None,
    ) -> "RVC":
        """Load `hubert.safetensors` and `rmvpe.safetensors` from `models_dir` (or explicit paths)."""
        if isinstance(dtype, str):
            dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype]
        hubert_path = hubert_path or _find(models_dir, _HUBERT_NAMES)
        rmvpe_path = rmvpe_path or _find(models_dir, _RMVPE_NAMES)
        return cls(HubertModel.from_pretrained(hubert_path, dtype=dtype), RMVPE.from_pretrained(rmvpe_path), config)

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        voice: Voice,
        pitch: float = 0,
        index_rate: float = 0.75,
        protect: float = 0.33,
        rms_mix_rate: float = 0.25,
        speaker_id: int = 0,
        f0_curve: Optional[np.ndarray] = None,
        output_sample_rate: Optional[int] = None,
        deterministic: bool = False,
        progress: Optional[Callable[[float], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert `audio` (mono or `(n, channels)` float array at `sample_rate`) to `voice`.

        :param pitch: transposition in semitones (e.g. +12 for male -> female).
        :param index_rate: 0..1, how strongly to pull features towards the voice's training data (needs an index).
        :param protect: 0..0.5, how much to protect consonants/breaths from retrieval artefacts (0.5 = off).
        :param rms_mix_rate: 0..1, 1 keeps the converted loudness, lower values follow the input's loudness.
        :param f0_curve: optional `(n, 2)` array of (seconds, Hz) overriding the detected pitch.
        :param output_sample_rate: resample the result (default: the voice's native rate).
        :param deterministic: disable the model's sampling noise (useful for tests / A-B comparisons).
        :returns: `(audio, sample_rate)`.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        x = resample(audio, sample_rate, 16000)
        pipe = Pipeline(voice.sample_rate, self.config, rmvpe=self.rmvpe)
        out = pipe.pipeline(
            self.hubert,
            voice,
            speaker_id,
            x,
            f0_up_key=pitch,
            index_rate=index_rate,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            inp_f0=f0_curve,
            deterministic=deterministic,
            progress=progress,
        )
        sr = voice.sample_rate
        if output_sample_rate and output_sample_rate != sr:
            out, sr = resample(out, sr, output_sample_rate), output_sample_rate
        return out, sr

    def convert_file(self, input_path: str, output_path: str, voice: Voice, **kwargs) -> Tuple[np.ndarray, int]:
        import soundfile as sf

        audio, sr = load_audio(input_path)
        out, out_sr = self.convert(audio, sr, voice, **kwargs)
        sf.write(output_path, out, out_sr)
        return out, out_sr
