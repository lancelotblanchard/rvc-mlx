"""
Command-line entry point for end-to-end RVC inference.

Usage:

    python -m rvc_mlx.infer \\
        --input speech.wav \\
        --output converted.wav \\
        --voice path/to/voice.pth \\
        --hubert path/to/hubert_base.pt \\
        --rmvpe path/to/rmvpe.pt

Or programmatically:

    from rvc_mlx.infer import run_inference
    run_inference("in.wav", "out.wav", voice="voice.pth", hubert="hubert_base.pt", rmvpe="rmvpe.pt")

Checkpoints in either the released `.pth` / `.pt` format or our `.safetensors` (with sibling `.config.json`) format
both work — `from_pretrained` auto-converts on first use.
"""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace
from typing import Optional

from rvc_mlx.audio import load_audio_16k, save_audio
from rvc_mlx.hubert import HubertModel
from rvc_mlx.pipeline import Pipeline
from rvc_mlx.rmvpe import RMVPE
from rvc_mlx.synthesizer import SynthesizerTrnMs768NSFsid

logger = logging.getLogger(__name__)


# Default pipeline-chunking parameters mirror RVC's upstream config. They control the segmentation strategy in
# `Pipeline.pipeline` and aren't tied to any particular voice/checkpoint.
_DEFAULT_PIPELINE_CONFIG = SimpleNamespace(
    x_pad=1,
    x_query=6,
    x_center=38,
    x_max=41,
    is_half=False,
    rmvpe_root=None,
    hubert_root=None,
)


def run_inference(
    input_path: str,
    output_path: str,
    voice: str,
    hubert: str,
    rmvpe: str,
    *,
    speaker_id: int = 0,
    pitch_shift: int = 0,
    rms_mix_rate: float = 0.25,
    hubert_output_layer: int = 12,
) -> str:
    """
    Run a full RVC inference: load audio, models, run the pipeline, write the result.

    :param input_path: path to a source audio file (any libsndfile-readable format).
    :param output_path: path for the generated audio (`.wav` by default).
    :param voice: per-voice synthesizer checkpoint (`.pth` released format or our `.safetensors`).
    :param hubert: HuBERT / ContentVec checkpoint.
    :param rmvpe: RMVPE pitch-extractor checkpoint.
    :param speaker_id: index into the synthesizer's speaker embedding table.
    :param pitch_shift: semitones to shift the source pitch by.
    :param rms_mix_rate: source-envelope transfer in `[0, 1]`. 0 = full transfer, 1 = pass-through. RVC default: 0.25.
    :param hubert_output_layer: which HuBERT transformer layer to read features from (12 for v2, 9 for v1).
    :returns: `output_path`.
    """
    logger.info("Loading HuBERT %s", hubert)
    hubert_model = HubertModel.from_pretrained(hubert)
    logger.info("Loading RMVPE %s", rmvpe)
    rmvpe_model = RMVPE.from_pretrained(rmvpe)
    logger.info("Loading synthesizer %s", voice)
    synthesizer = SynthesizerTrnMs768NSFsid.from_pretrained(voice)

    pipe = Pipeline(
        tgt_sr=synthesizer.sr,
        config=_DEFAULT_PIPELINE_CONFIG,
        rmvpe=rmvpe_model,
        hubert=hubert_model,
    )

    logger.info("Loading input audio %s", input_path)
    audio = load_audio_16k(input_path)

    logger.info(
        "Running pipeline (sid=%d, f0_up_key=%d, rms_mix_rate=%.2f, hubert_layer=%d)",
        speaker_id,
        pitch_shift,
        rms_mix_rate,
        hubert_output_layer,
    )
    audio_out = pipe.pipeline(
        synthesizer,
        audio,
        sid=speaker_id,
        f0_up_key=pitch_shift,
        output_layer=hubert_output_layer,
        rms_mix_rate=rms_mix_rate,
    )

    logger.info("Writing %s (%d samples @ %d Hz)", output_path, audio_out.shape[0], synthesizer.sr)
    save_audio(output_path, audio_out, synthesizer.sr)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rvc-mlx",
        description="Run RVC voice conversion in MLX.",
    )
    p.add_argument("--input", "-i", required=True, help="Source audio file (WAV/FLAC/OGG/...).")
    p.add_argument("--output", "-o", required=True, help="Output audio file (WAV recommended).")
    p.add_argument("--voice", required=True, help="Per-voice synthesizer checkpoint (.pth or .safetensors).")
    p.add_argument("--hubert", required=True, help="HuBERT / ContentVec checkpoint (.pt or .safetensors).")
    p.add_argument("--rmvpe", required=True, help="RMVPE pitch-extractor checkpoint (.pt or .safetensors).")
    p.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Speaker index into the synthesizer's embedding table (default: 0).",
    )
    p.add_argument(
        "--pitch-shift",
        type=int,
        default=0,
        help="Pitch shift in semitones; positive shifts up (default: 0).",
    )
    p.add_argument(
        "--rms-mix-rate",
        type=float,
        default=0.25,
        help="Source-envelope transfer in [0, 1]; 0 = full transfer, 1 = pass-through (default: 0.25).",
    )
    p.add_argument(
        "--hubert-layer",
        type=int,
        default=12,
        help="HuBERT transformer layer to extract content from (default: 12 for ContentVec v2; use 9 for v1).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable info-level logging from the pipeline.",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_inference(
        input_path=args.input,
        output_path=args.output,
        voice=args.voice,
        hubert=args.hubert,
        rmvpe=args.rmvpe,
        speaker_id=args.speaker_id,
        pitch_shift=args.pitch_shift,
        rms_mix_rate=args.rms_mix_rate,
        hubert_output_layer=args.hubert_layer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
