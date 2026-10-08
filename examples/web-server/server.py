"""
A tiny local web UI for rvc-mlx.

    pip install "rvc-mlx[server]"
    python examples/web-server/server.py --models models/

then open http://127.0.0.1:8000. Voices are the `.safetensors` files in `<models>/voices/` (rescanned on every request,
so you can drop new ones in while the server runs).
"""

from __future__ import annotations

import argparse
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from rvc_mlx import RVC, Voice
from rvc_mlx.io import read_metadata

STATIC = Path(__file__).parent / "static"


def create_app(models_dir: str, rvc: RVC | None = None) -> FastAPI:
    app = FastAPI(title="rvc-mlx")
    voices_dir = Path(models_dir) / "voices"
    state = {"rvc": rvc}
    cache: dict[str, tuple[float, Voice]] = {}
    lock = threading.Lock()  # one conversion at a time: MLX already uses the whole GPU

    def engine() -> RVC:
        if state["rvc"] is None:
            state["rvc"] = RVC.from_pretrained(models_dir)
        return state["rvc"]

    def voice_files() -> dict[str, Path]:
        if not voices_dir.is_dir():
            return {}
        return {p.stem: p for p in sorted(voices_dir.glob("*.safetensors"))}

    def load_voice(voice_id: str) -> Voice:
        path = voice_files().get(voice_id)
        if path is None:
            raise HTTPException(404, f"Unknown voice {voice_id!r}")
        mtime = path.stat().st_mtime
        hit = cache.get(voice_id)
        if hit is None or hit[0] != mtime:
            cache[voice_id] = (mtime, Voice.load(str(path)))
        return cache[voice_id][1]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/voices")
    def voices():
        out = []
        for voice_id, path in voice_files().items():
            meta = read_metadata(str(path))
            if meta.get("kind") != "voice":
                continue
            out.append(
                {
                    "id": voice_id,
                    "name": meta.get("name", voice_id),
                    "version": meta.get("version"),
                    "sample_rate": int(meta.get("sample_rate", 0)),
                    "pitch": meta.get("f0") == "1",
                    "index": int(meta.get("index_size", 0)) > 0,
                    "blend": meta.get("merged_from"),
                }
            )
        return out

    @app.post("/api/convert")
    def convert(
        audio: UploadFile = File(...),
        voice: str = Form(...),
        pitch: float = Form(0),
        index_rate: float = Form(0.75),
        protect: float = Form(0.33),
        rms_mix_rate: float = Form(0.25),
    ):
        try:
            data, sr = sf.read(io.BytesIO(audio.file.read()), dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001 — report any decode error to the client
            raise HTTPException(400, f"Could not read audio ({e}). Send WAV/FLAC/OGG.") from e
        if data.shape[0] < sr // 4:
            raise HTTPException(400, "Audio is shorter than 0.25 s")
        v = load_voice(voice)
        with lock:
            t0 = time.perf_counter()
            out, out_sr = engine().convert(
                data, sr, v, pitch=pitch, index_rate=index_rate, protect=protect, rms_mix_rate=rms_mix_rate
            )
            elapsed = time.perf_counter() - t0
        buf = io.BytesIO()
        sf.write(buf, out, out_sr, format="WAV", subtype="PCM_16")
        return Response(
            buf.getvalue(),
            media_type="audio/wav",
            headers={"X-Elapsed": f"{elapsed:.3f}", "X-Duration": f"{len(out) / out_sr:.3f}"},
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="rvc-mlx local web UI")
    parser.add_argument("--models", default="models", help="directory with hubert/rmvpe .safetensors and voices/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not os.path.isdir(args.models):
        raise SystemExit(f"{args.models!r} not found. See docs/converting-models.md.")
    rvc = RVC.from_pretrained(args.models)  # load once up front so the first request is fast
    print(f"Serving {args.models} on http://{args.host}:{args.port}")
    uvicorn.run(create_app(args.models, rvc), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
