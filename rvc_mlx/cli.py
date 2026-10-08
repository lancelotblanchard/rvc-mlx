"""
`rvc-mlx` command line.

    rvc-mlx convert-base --hubert hubert_base.pt --rmvpe rmvpe.pt -o models/
    rvc-mlx convert-voice alice.pth --index added_IVF256_Flat_nprobe_1_alice_v2.index -o models/voices/
    rvc-mlx infer -m models/ -v models/voices/alice.safetensors input.wav output.wav --pitch 12
    rvc-mlx blend models/voices/alice.safetensors:0.7 models/voices/bob.safetensors:0.3 -o models/voices/mix.safetensors
    rvc-mlx info models/voices/alice.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _out_path(src: str, out: str | None, default_name: str | None = None) -> str:
    name = default_name or os.path.splitext(os.path.basename(src))[0] + ".safetensors"
    if out is None:
        return os.path.join(os.path.dirname(src) or ".", name)
    if out.endswith(".safetensors"):
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        return out
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, name)


def cmd_convert_base(args) -> None:
    from rvc_mlx.convert import convert_hubert_checkpoint, convert_rmvpe_checkpoint

    if not (args.hubert or args.rmvpe):
        sys.exit("Pass --hubert and/or --rmvpe")
    if args.hubert:
        p = convert_hubert_checkpoint(args.hubert, _out_path(args.hubert, args.out, "hubert.safetensors"), dtype=args.dtype)
        print(f"hubert -> {p}")
    if args.rmvpe:
        p = convert_rmvpe_checkpoint(args.rmvpe, _out_path(args.rmvpe, args.out, "rmvpe.safetensors"), dtype=args.dtype)
        print(f"rmvpe  -> {p}")


def cmd_convert_voice(args) -> None:
    from rvc_mlx.convert import convert_voice_checkpoint
    from rvc_mlx.voice import Voice

    p = convert_voice_checkpoint(
        args.pth,
        _out_path(args.pth, args.out),
        index_path=args.index,
        name=args.name,
        dtype=args.dtype,
        max_index_vectors=args.max_index_vectors,
    )
    print(f"{Voice.load(p)} -> {p}")


def cmd_infer(args) -> None:
    import mlx.core as mx

    from rvc_mlx.api import RVC
    from rvc_mlx.voice import Voice

    t0 = time.perf_counter()
    rvc = RVC.from_pretrained(args.models, dtype=args.dtype)
    voice = Voice.load(args.voice, dtype=mx.float16 if args.dtype == "float16" else mx.float32)
    t1 = time.perf_counter()
    out, sr = rvc.convert_file(
        args.input,
        args.output,
        voice,
        pitch=args.pitch,
        index_rate=args.index_rate,
        protect=args.protect,
        rms_mix_rate=args.rms_mix_rate,
        speaker_id=args.speaker,
        output_sample_rate=args.sample_rate,
    )
    t2 = time.perf_counter()
    print(f"{args.output}: {len(out) / sr:.2f} s of audio in {t2 - t1:.2f} s (models loaded in {t1 - t0:.2f} s)")


def cmd_blend(args) -> None:
    from rvc_mlx.voice import Voice, blend_voices

    voices, weights = [], []
    for spec in args.voices:
        path, sep, w = spec.rpartition(":")
        if not sep or not w.replace(".", "", 1).isdigit():
            path, w = spec, "1"
        voices.append(Voice.load(path))
        weights.append(float(w))
    blended = blend_voices(voices, weights, name=args.name)
    blended.save(args.out)
    print(f"{blended} -> {args.out}  ({blended.metadata['merged_from']})")


def cmd_info(args) -> None:
    from rvc_mlx.io import read_metadata

    meta = read_metadata(args.path)
    if "config" in meta:
        meta["config"] = json.loads(meta["config"])
    print(json.dumps(meta, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="rvc-mlx", description="RVC voice conversion on Apple silicon with MLX")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert-base", help="convert hubert_base.pt and rmvpe.pt (needs torch)")
    p.add_argument("--hubert", help="hubert_base.pt (fairseq) or a HF ContentVec pytorch_model.bin / .safetensors")
    p.add_argument("--rmvpe", help="rmvpe.pt")
    p.add_argument("-o", "--out", default="models", help="output directory (default: models/)")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p.set_defaults(func=cmd_convert_base)

    p = sub.add_parser("convert-voice", help="convert an RVC voice .pth (+ .index) (needs torch; faiss for .index)")
    p.add_argument("pth")
    p.add_argument("--index", help="the voice's added_*.index (or total_fea.npy)")
    p.add_argument("-o", "--out", help="output file or directory (default: next to the .pth)")
    p.add_argument("--name", help="display name (default: file name)")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--max-index-vectors", type=int, help="k-means the feature bank down to this size (e.g. 20000 for phones)")
    p.set_defaults(func=cmd_convert_voice)

    p = sub.add_parser("infer", help="convert an audio file")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("-m", "--models", default="models", help="directory with hubert/rmvpe .safetensors")
    p.add_argument("-v", "--voice", required=True, help="converted voice .safetensors")
    p.add_argument("-p", "--pitch", type=float, default=0, help="semitones")
    p.add_argument("--index-rate", type=float, default=0.75)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--rms-mix-rate", type=float, default=0.25)
    p.add_argument("--speaker", type=int, default=0)
    p.add_argument("--sample-rate", type=int, help="output sample rate (default: the voice's)")
    p.add_argument("--dtype", default="float32", choices=["float16", "float32"], help="compute precision")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("blend", help="interpolate the weights of compatible voices")
    p.add_argument("voices", nargs="+", help="path[:weight] ...")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--name")
    p.set_defaults(func=cmd_blend)

    p = sub.add_parser("info", help="print a converted file's metadata")
    p.add_argument("path")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
