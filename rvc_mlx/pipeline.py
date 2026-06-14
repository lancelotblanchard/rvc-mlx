import logging
from typing import Optional

import mlx.core as mx
import numpy as np

from rvc_mlx.hubert import HubertModel
from rvc_mlx.rmvpe import RMVPE
from rvc_mlx.synthesizer import SynthesizerTrnMs768NSFsid
from rvc_mlx.utils import interpolate_nearest_axis

logger = logging.getLogger(__name__)


class Pipeline(object):
    """
    End-to-end RVC voice-conversion pipeline.

    Wires the three learned models together:
      * `HubertModel` for per-frame content extraction (50 Hz frame rate)
      * `RMVPE` for per-frame f0 extraction (100 Hz frame rate)
      * `SynthesizerTrnMs768NSFsid` for the final voice-converted audio

    `vc` is the inference entry point: pass raw 16 kHz audio + a loaded synthesizer + a speaker id and get audio back
    at the synthesizer's target sample rate.

    The synthesizer is supplied at call time (not loaded in `__init__`) because it's per-voice: a single pipeline
    instance is typically reused across voices, swapping the synthesizer for each.
    """

    def __init__(
        self,
        tgt_sr: int,
        config,
        *,
        rmvpe: Optional[RMVPE] = None,
        hubert: Optional[HubertModel] = None,
    ):
        """
        :param tgt_sr: target output sample rate (matches the synthesizer's `sr`).
        :param config: object with the RVC pipeline attributes (`x_pad`, `x_query`, `x_center`, `x_max`, `is_half`,
            `rmvpe_root`, `hubert_root`). Only `is_half` and the root paths are used at construction; the rest are
            kept for compatibility with the upstream pipeline shape but aren't consumed by `vc` in this minimal port.
        :param rmvpe: optional pre-loaded `RMVPE` instance. If `None`, loaded from `config.rmvpe_root/rmvpe.pt`.
        :param hubert: optional pre-loaded `HubertModel` instance. If `None`, loaded from
            `config.hubert_root/hubert_base.pt`.
        """
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.tgt_sr = tgt_sr

        # Input sample rate is fixed at 16 kHz: HuBERT and RMVPE both expect this. The synthesizer's output runs at
        # `tgt_sr` (32k / 40k / 48k depending on the voice).
        self.sr = 16_000
        # Note: `window` here mirrors the upstream RVC pipeline value used for `inp_f0` splicing inside `get_f0`,
        # NOT the actual f0 hop length. The latter is `f0_hop = 160` below.
        self.window = 16
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        # HuBERT's frame rate is sr / 320 = 50 Hz. RMVPE's frame rate is sr / 160 = 100 Hz. The pipeline upsamples
        # HuBERT features 2x to align with the f0 grid.
        self.hubert_hop = 320
        self.f0_hop = 160

        if rmvpe is None:
            logger.info(f"Loading RMVPE model {config.rmvpe_root}/rmvpe.pt")
            rmvpe = RMVPE.from_pretrained(
                f"{config.rmvpe_root}/rmvpe.pt",
                is_half=self.is_half,
            )
        if hubert is None:
            logger.info(f"Loading HuBERT model {config.hubert_root}/hubert_base.pt")
            hubert = HubertModel.from_pretrained(
                f"{config.hubert_root}/hubert_base.pt",
            )

        self.model_rmvpe = rmvpe
        self.hubert = hubert

    def get_f0(self, x: np.ndarray, f0_up_key: int, inp_f0: Optional[np.ndarray] = None):
        """
        Extract per-frame pitch from raw audio.

        :param x: 1D numpy array of raw 16 kHz audio.
        :param f0_up_key: pitch shift in semitones (positive = higher).
        :param inp_f0: optional user-supplied f0 override of shape `(N, 2)` where column 0 is time in seconds and
            column 1 is f0 in Hz. Used to splice an externally-edited pitch contour into a slice of the auto-extracted
            f0. The slice runs from `x_pad * (sr/window)` for the supplied duration.
        :returns: `(f0_coarse, f0bak)` where `f0_coarse` is integer pitch class indices in `[1, 255]` for the
            synthesizer's pitch embedding and `f0bak` is the continuous f0 in Hz for the NSF source module.
        """
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        f0 = self.model_rmvpe.infer_from_audio(mx.array(x), thred=0.03)

        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window  # f0 frames per second
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (
            (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        )
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak

    def get_features(
        self,
        audio: mx.array,
        output_layer: int = 12,
    ) -> mx.array:
        """
        Extract HuBERT content features from raw audio, upsampled to the f0 frame rate.

        :param audio: raw 16 kHz audio. Accepts shape `(T,)` or `(B, T)`.
        :param output_layer: 1-indexed HuBERT transformer layer to read from. Use 12 for ContentVec v2
            (RVC v2 default), 9 for HuBERT-base v1.
        :returns: features of shape `(B, T_features, 768)` at the f0 frame rate (100 Hz). Run through a 2x
            nearest-neighbour upsample along the time axis to match the RMVPE f0 grid.
        """
        if audio.ndim == 1:
            audio = mx.expand_dims(audio, 0)
        feats = self.hubert.extract_features(audio, output_layer=output_layer)
        # Upsample 2x along the time axis: HuBERT runs at sr/320 = 50 Hz; f0 runs at sr/160 = 100 Hz.
        feats = interpolate_nearest_axis(feats, 2, axis=1)
        return feats

    def vc(
        self,
        synthesizer: SynthesizerTrnMs768NSFsid,
        audio: mx.array,
        sid: int,
        f0_up_key: int = 0,
        output_layer: int = 12,
        *,
        noise_z: Optional[mx.array] = None,
        rand_ini: Optional[mx.array] = None,
        noise_raw: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Run full RVC inference: audio -> phone -> f0 -> synthesizer -> audio.

        :param synthesizer: a loaded `SynthesizerTrnMs768NSFsid` for the target voice.
        :param audio: 1D `mx.array` of raw 16 kHz audio.
        :param sid: integer speaker index (into the synthesizer's speaker-embedding table).
        :param f0_up_key: pitch shift in semitones.
        :param output_layer: HuBERT transformer layer to extract content from (12 for v2, 9 for v1).
        :param noise_z / rand_ini / noise_raw: forwarded to `synthesizer.infer`. Optional; if `None`, the synthesizer
            samples its own random tensors. Provide explicit values for deterministic output.
        :returns: 1D `mx.array` of generated audio at `synthesizer.sr`.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected a 1D audio array at 16 kHz, got shape {audio.shape}.")

        # Run RMVPE (numpy-based decode step internally). RMVPE returns (T_f0,) at 100 Hz.
        audio_np = np.array(audio)
        f0_coarse_np, f0bak_np = self.get_f0(audio_np, f0_up_key)

        # Extract HuBERT features and upsample to the f0 grid.
        feats = self.get_features(audio, output_layer=output_layer)  # (1, T_features, 768)

        # Clip both sequences to the shorter of the two so they line up frame-for-frame.
        p_len = int(min(feats.shape[1], len(f0_coarse_np)))
        feats = feats[:, :p_len, :]
        pitch = mx.array(f0_coarse_np[:p_len], dtype=mx.int64).reshape(1, p_len)
        pitchf = mx.array(f0bak_np[:p_len].astype(np.float32)).reshape(1, p_len)
        phone_lengths = mx.array([p_len], dtype=mx.int64)
        sid_arr = mx.array([sid], dtype=mx.int64)

        o, _, _ = synthesizer.infer(
            feats,
            phone_lengths,
            pitch,
            pitchf,
            sid_arr,
            noise_z=noise_z,
            rand_ini=rand_ini,
            noise_raw=noise_raw,
        )
        # `o` has shape (1, T_audio_out, 1) in MLX channels-last. Squeeze to 1D for the caller.
        return mx.squeeze(mx.squeeze(o, axis=0), axis=-1)
