import logging
from typing import List, Optional, Tuple

import librosa
import mlx.core as mx
import numpy as np

from rvc_mlx.hubert import HubertModel
from rvc_mlx.rmvpe import RMVPE
from rvc_mlx.synthesizer import SynthesizerTrnMs768NSFsid
from rvc_mlx.utils import interpolate_nearest_axis

logger = logging.getLogger(__name__)


def change_rms(
    data1: np.ndarray,
    sr1: int,
    data2: np.ndarray,
    sr2: int,
    rate: float,
) -> np.ndarray:
    """
    Rescale `data2`'s amplitude envelope to a blend of `data1`'s and `data2`'s loudness contours.

    Per ~half-second window, compute RMS of each signal, linearly interpolate both onto `data2`'s sample grid, then
    multiply `data2` by `rms1 ** (1 - rate) * rms2 ** (rate - 1)`. Conceptually:
      * `rate=0` -> output is rescaled so its envelope matches the input (full transfer of the source's dynamics).
      * `rate=1` -> output passes through unchanged.
      * In between, the two envelopes are interpolated in log-amplitude space.

    Mirrors the RVC reference's `change_rms` but uses NumPy throughout (the reference temporarily moves through
    PyTorch for the `F.interpolate(mode="linear")` step).

    :param data1: source audio (the loudness reference).
    :param sr1: sample rate of `data1`.
    :param data2: generated audio to rescale.
    :param sr2: sample rate of `data2`.
    :param rate: blending factor in `[0, 1]`. RVC's default is `0.25` (mostly transfer the source loudness).
    :returns: a new array, same shape and dtype as `data2`, with the rescaled envelope.
    """
    # Half-second analysis windows in each signal's own time domain.
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )[0]
    rms2 = librosa.feature.rms(
        y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2
    )[0]
    # Interpolate each RMS envelope onto data2's full per-sample grid.
    target = np.arange(data2.shape[0])
    rms1_full = np.interp(target, np.linspace(0, data2.shape[0] - 1, len(rms1)), rms1)
    rms2_full = np.interp(target, np.linspace(0, data2.shape[0] - 1, len(rms2)), rms2)
    # Avoid division by zero in silent regions.
    rms2_full = np.maximum(rms2_full, 1e-6)
    scale = np.power(rms1_full, 1.0 - rate) * np.power(rms2_full, rate - 1.0)
    return (data2 * scale).astype(data2.dtype)


def find_split_points(
    audio: np.ndarray, window: int, t_query: int, t_center: int, t_max: int
) -> List[int]:
    """
    Find sample indices at which to split a long audio clip into shorter segments, choosing low-energy spots so the
    boundary artifacts are minimized.

    Algorithm (mirrors the RVC reference's `pipeline` segmentation block):
      1. Compute a windowed sum-of-absolute-values envelope over `window` samples.
      2. For each anchor at `t_center, 2 * t_center, ...` along the time axis, search a `±t_query`-sample neighborhood
         for the minimum of the envelope and pick that index as a split point.

    :param audio: 1D padded audio array (the same one fed into `vc`).
    :param window: stride used for f0/feature framing, in samples (typically the f0 hop, `f0_hop = 160`).
    :param t_query: search radius around each anchor, in samples.
    :param t_center: anchor stride, in samples.
    :param t_max: only segment if the audio is longer than this many samples.
    :returns: sorted list of split sample indices. Empty list if no segmentation is needed.
    """
    if audio.shape[0] <= t_max:
        return []

    # Sum-of-abs across a sliding `window`-sample frame: a cheap energy proxy that's robust to phase / sign.
    audio_sum = np.zeros_like(audio)
    for i in range(window):
        # Each shift adds another sample's abs value to the sum at every position; the end of the array is implicitly
        # zero-padded by the `i:i - window` slice (which has length `len(audio) - window`).
        audio_sum[: audio.shape[0] - window] += np.abs(audio[i : i + audio.shape[0] - window])

    splits: List[int] = []
    for t in range(t_center, audio.shape[0], t_center):
        lo = max(t - t_query, 0)
        hi = min(t + t_query, audio.shape[0])
        window_slice = audio_sum[lo:hi]
        if window_slice.size == 0:
            continue
        # Local minimum: the position with the lowest sum-of-abs energy. `argmin` returns the first such index when
        # there are ties; align it back to the absolute sample index.
        splits.append(lo + int(np.argmin(window_slice)))
    return splits


class Pipeline(object):
    """
    End-to-end RVC voice-conversion pipeline.

    Wires the three learned models together:
      * `HubertModel` for per-frame content extraction (50 Hz frame rate)
      * `RMVPE` for per-frame f0 extraction (100 Hz frame rate)
      * `SynthesizerTrnMs768NSFsid` for the final voice-converted audio

    Two entry points:
      * `vc(synthesizer, audio, sid, ...)` is the minimal one-shot conversion (no segmentation, no RMS transfer).
        Suitable for short clips that fit in memory.
      * `pipeline(synthesizer, audio_np, sid, ...)` is the full upstream-RVC-style flow: reflect-pad, segment at
        low-energy points, run `vc` on each segment, concatenate, optionally apply RMS rescaling. This is the entry
        point you want for arbitrary-length input.

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

    def pipeline(
        self,
        synthesizer: SynthesizerTrnMs768NSFsid,
        audio: np.ndarray,
        sid: int = 0,
        f0_up_key: int = 0,
        output_layer: int = 12,
        rms_mix_rate: float = 0.25,
        protect: float = 0.5,
    ) -> np.ndarray:
        """
        Full RVC pipeline: arbitrary-length 16 kHz audio in, voice-converted audio out at `synthesizer.sr`.

        Mirrors the inference path of the upstream RVC `Pipeline.pipeline` method:
          1. Reflect-pad the input by `t_pad` samples on each side so the boundary frames have context.
          2. If the padded audio is longer than `t_max` samples, find low-energy split points and chunk the audio.
          3. Run `vc` on each chunk independently, then concatenate the outputs.
          4. Strip the padding (`t_pad_tgt` samples on each side, scaled to the synthesizer's output rate).
          5. Optionally apply RMS rescaling so the output's loudness envelope tracks the source.

        :param synthesizer: target-voice `SynthesizerTrnMs768NSFsid`.
        :param audio: 1D numpy array at 16 kHz (any length).
        :param sid: integer speaker index into the synthesizer's embedding table.
        :param f0_up_key: pitch shift in semitones.
        :param output_layer: HuBERT layer to extract from (12 for v2, 9 for v1).
        :param rms_mix_rate: RMS rescaling blend in `[0, 1]`. `0` = full source-envelope transfer, `1` = pass-through.
            RVC's default is `0.25`.
        :param protect: currently unused (kept in the signature for API compatibility with upstream RVC's
            `pipeline`; the consonants-protection feature isn't implemented in this port).
        :returns: 1D numpy array at the synthesizer's sample rate.
        """
        del protect  # not implemented in this port; documented above.

        if audio.ndim != 1:
            raise ValueError(f"Expected a 1D audio array at 16 kHz, got shape {audio.shape}.")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Reflect-pad both ends so boundary frames see real audio context, matching the upstream RVC behavior.
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")

        # Segment at low-energy points if the clip is longer than t_max.
        splits = find_split_points(
            audio_pad,
            window=self.f0_hop,
            t_query=self.t_query,
            t_center=self.t_center,
            t_max=self.t_max,
        )

        # Build the list of (start, end) sample ranges to process. Each segment is processed independently with
        # `vc`, and we crop `t_pad_tgt` samples off each end of every chunk's output to drop the padded context.
        ranges: List[Tuple[int, int]] = []
        s = 0
        for t in splits:
            # Snap each split to a window boundary so per-segment frame counts line up.
            t = (t // self.f0_hop) * self.f0_hop
            ranges.append((s, t + self.f0_hop * 2))  # extra hop overlap so the boundary frame is shared
            s = t
        ranges.append((s, audio_pad.shape[0]))

        # Compute the upsample factor between input audio (16 kHz) and synthesizer output. `pipeline` runs at the
        # synthesizer's sample rate, so the crop length needs to scale accordingly.
        sr_ratio = self.tgt_sr / self.sr
        t_pad_tgt = int(self.t_pad * sr_ratio)

        out_segments: List[np.ndarray] = []
        for seg_idx, (lo, hi) in enumerate(ranges):
            chunk = audio_pad[lo:hi]
            if chunk.size == 0:
                continue
            chunk_mx = mx.array(chunk)
            out_mx = self.vc(synthesizer, chunk_mx, sid=sid, f0_up_key=f0_up_key, output_layer=output_layer)
            out_np = np.array(out_mx).astype(np.float32)
            # For interior segments, crop t_pad_tgt off both sides. For the boundary segments (first/last), only
            # crop the side that abuts the original padding.
            crop_left = t_pad_tgt if seg_idx > 0 else t_pad_tgt
            crop_right = t_pad_tgt if seg_idx < len(ranges) - 1 else t_pad_tgt
            if crop_left + crop_right >= out_np.shape[0]:
                # Defensive: if the segment is so short that the crops would overlap, skip cropping for safety.
                out_segments.append(out_np)
            else:
                out_segments.append(out_np[crop_left:-crop_right] if crop_right > 0 else out_np[crop_left:])

        audio_out = np.concatenate(out_segments) if out_segments else np.zeros((0,), dtype=np.float32)

        if rms_mix_rate < 1.0 and audio_out.size > 0:
            audio_out = change_rms(audio, self.sr, audio_out, self.tgt_sr, rms_mix_rate)

        return audio_out
