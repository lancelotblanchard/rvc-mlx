"""
The RVC voice-conversion pipeline (a port of the WebUI's `infer/modules/vc/pipeline.py`).

Given 16 kHz input audio and a voice, the pipeline:

1. high-passes the input (48 Hz Butterworth, zero phase) and reflect-pads it by `x_pad` seconds;
2. estimates f0 with RMVPE (100 frames/s), applies the pitch shift and quantises it to 255 coarse bins;
3. for long inputs, splits at the quietest points near every `x_center` seconds so each chunk fits in memory;
4. per chunk: extracts HuBERT features (50 frames/s), optionally blends them with their nearest neighbours in the
   voice's feature bank (`index_rate`), upsamples them 2x to the f0 frame rate, protects unvoiced consonants by
   mixing back un-retrieved features (`protect`), and synthesizes audio at the voice's sample rate;
5. stitches the chunks, optionally mixes the input's loudness envelope back in (`rms_mix_rate`) and peak-limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import mlx.core as mx
import numpy as np

from rvc_mlx.dsp import change_rms, highpass
from rvc_mlx.index import retrieve_blended
from rvc_mlx.rmvpe import RMVPE

logger = logging.getLogger(__name__)

F0_MIN = 50
F0_MAX = 1100
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


@dataclass
class PipelineConfig:
    """
    Chunking geometry, in seconds. Defaults match the WebUI's full-precision settings; inputs longer than `x_max` are
    cut near every `x_center` seconds (searching ±`x_query` for silence) with `x_pad` seconds of context per side.
    """

    x_pad: int = 1
    x_query: int = 6
    x_center: int = 38
    x_max: int = 41
    is_half: bool = False
    rmvpe_root: str = "."


def coarse_f0(f0: np.ndarray) -> np.ndarray:
    """Quantise f0 (Hz) to RVC's 1..255 mel-spaced bins (1 = unvoiced / out of range low)."""
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(np.int32)


class Pipeline(object):
    def __init__(self, tgt_sr, config, rmvpe: Optional[RMVPE] = None):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16_000
        self.window = 160  # samples per f0 frame (RMVPE hop length at 16 kHz)
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max

        if rmvpe is None:
            logger.info(f"Loading RMVPE model {config.rmvpe_root}/rmvpe.pt")
            # `from_pretrained` accepts either a `.safetensors` file or the original RVC `.pt`. For `.pt` it
            # auto-converts to a sibling `.safetensors` on first use (requires torch installed).
            rmvpe = RMVPE.from_pretrained(f"{config.rmvpe_root}/rmvpe.pt", is_half=self.is_half)
        self.model_rmvpe = rmvpe

    # ---------------------------------------------------------------------------------------------------------- f0
    def get_f0(self, x, f0_up_key, inp_f0=None):
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window  # f0 per second
        if inp_f0 is not None:
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]

        f0bak = f0.copy()
        return coarse_f0(f0), f0bak

    # ---------------------------------------------------------------------------------------------------------- vc
    def vc(
        self,
        hubert,
        voice,
        sid: int,
        audio0: np.ndarray,
        pitch: Optional[np.ndarray],
        pitchf: Optional[np.ndarray],
        index_rate: float,
        protect: float,
        speaker_embedding: Optional[mx.array] = None,
        deterministic: bool = False,
        index_banks: Optional[Sequence[mx.array]] = None,
        index_weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Convert one chunk. Returns float32 audio at the voice's sample rate (including the padding)."""
        dtype = voice.model.enc_p.emb_phone.weight.dtype
        if voice.has_f0 and (pitch is None or pitchf is None):
            raise ValueError(f"Voice {voice.name!r} is pitch-conditioned; pass `pitch` and `pitchf`.")
        feats = hubert(mx.array(audio0.astype(np.float32))[None], version=voice.version).astype(mx.float32)
        has_pitch = pitch is not None and pitchf is not None
        if protect < 0.5 and has_pitch:
            feats0 = feats

        banks = list(index_banks) if index_banks is not None else ([voice.index] if voice.index is not None else [])
        if banks and index_rate != 0:
            feats = retrieve_blended(feats, banks, index_weights) * index_rate + (1 - index_rate) * feats

        feats = mx.repeat(feats, 2, axis=1)  # 50 -> 100 frames/s (nearest), matching F.interpolate(scale_factor=2)
        if protect < 0.5 and has_pitch:
            feats0 = mx.repeat(feats0, 2, axis=1)

        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
        feats = feats[:, :p_len]
        if has_pitch:
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]

        if protect < 0.5 and has_pitch:
            # Voiced frames keep the retrieved features; unvoiced ones (consonants, breaths) lean on the originals.
            pitchff = np.where(pitchf < 1, protect, 1.0).astype(np.float32)[..., None]
            pitchff = mx.array(pitchff)
            feats = feats * pitchff + feats0[:, :p_len] * (1 - pitchff)

        noise = dict(noise_scale=0.0, nsf_noise_scale=0.0) if deterministic else {}
        audio1 = voice.model.infer(
            feats.astype(dtype),
            mx.array([p_len]),
            mx.array(pitch) if has_pitch else None,
            mx.array(pitchf.astype(np.float32)) if has_pitch else None,
            mx.array([sid]),
            speaker_embedding=speaker_embedding,
            **noise,
        )
        return np.array(audio1[0].astype(mx.float32))

    # ---------------------------------------------------------------------------------------------------- pipeline
    def split_points(self, audio: np.ndarray) -> List[int]:
        """Chunk boundaries: the quietest sample within ±x_query s of every x_center s (RVC's `opt_ts`)."""
        opt_ts: List[int] = []
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                region = audio_sum[t - self.t_query : t + self.t_query]
                opt_ts.append(t - self.t_query + np.where(region == region.min())[0][0])
        return opt_ts

    def pipeline(
        self,
        hubert,
        voice,
        sid: int,
        audio: np.ndarray,
        f0_up_key: float = 0,
        index_rate: float = 0.75,
        tgt_sr: Optional[int] = None,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        inp_f0: Optional[np.ndarray] = None,
        speaker_embedding: Optional[mx.array] = None,
        deterministic: bool = False,
        index_banks: Optional[Sequence[mx.array]] = None,
        index_weights: Optional[Sequence[float]] = None,
        progress: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """
        Convert a whole 16 kHz mono float signal. Returns float32 audio at `voice.sample_rate`, peak-limited to 0.99.
        """
        tgt_sr = tgt_sr or voice.sample_rate
        audio = highpass(np.asarray(audio, dtype=np.float64))
        opt_ts = self.split_points(audio)
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window

        pitch = pitchf = None
        if voice.has_f0:
            pitch, pitchf = self.get_f0(audio_pad, f0_up_key, inp_f0)
            pitch = pitch[:p_len][None].astype(np.int32)
            pitchf = pitchf[:p_len][None].astype(np.float32)

        common = dict(
            speaker_embedding=speaker_embedding,
            deterministic=deterministic,
            index_banks=index_banks,
            index_weights=index_weights,
        )
        audio_opt = []
        s = 0
        t = None
        n_chunks = len(opt_ts) + 1
        for i, t in enumerate(opt_ts):
            t = t // self.window * self.window
            sl = slice(s // self.window, (t + self.t_pad2) // self.window)
            out = self.vc(
                hubert, voice, sid, audio_pad[s : t + self.t_pad2 + self.window],
                pitch[:, sl] if pitch is not None else None,
                pitchf[:, sl] if pitchf is not None else None,
                index_rate, protect, **common,
            )
            audio_opt.append(out[self.t_pad_tgt : -self.t_pad_tgt])
            s = t
            if progress:
                progress((i + 1) / n_chunks)
        tail = slice(t // self.window, None) if t is not None else slice(None)
        out = self.vc(
            hubert, voice, sid, audio_pad[t:] if t is not None else audio_pad,
            pitch[:, tail] if pitch is not None else None,
            pitchf[:, tail] if pitchf is not None else None,
            index_rate, protect, **common,
        )
        audio_opt.append(out[self.t_pad_tgt : -self.t_pad_tgt])
        if progress:
            progress(1.0)

        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        peak = np.abs(audio_opt).max() / 0.99 if audio_opt.size else 0
        if peak > 1:
            audio_opt = audio_opt / peak
        return audio_opt.astype(np.float32)
