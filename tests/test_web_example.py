"""Smoke test for examples/web-server: voice listing, conversion round trip, error handling."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from rvc_mlx import RVC  # noqa: E402
from rvc_mlx.convert import convert_voice_checkpoint  # noqa: E402
from rvc_mlx.pipeline import PipelineConfig  # noqa: E402
from tests.test_cli import workspace  # noqa: E402,F401 — reuse the tiny-model fixture
from tests.test_pipeline_e2e import _small_rmvpe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _server_module():
    spec = importlib.util.spec_from_file_location("rvc_web_server", ROOT / "examples" / "web-server" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def client(workspace):  # noqa: F811
    from rvc_mlx.convert import convert_hubert_checkpoint
    from rvc_mlx.hubert import HubertModel

    models = workspace / "models"
    (models / "voices").mkdir(parents=True)
    convert_voice_checkpoint(str(workspace / "alice.pth"), str(models / "voices" / "alice.safetensors"),
                             index_path=str(workspace / "alice_fea.npy"), name="Alice")
    hubert = HubertModel.from_pretrained(convert_hubert_checkpoint(str(workspace / "hubert_base.pt"), str(models / "hubert.safetensors")))
    rvc = RVC(hubert, _small_rmvpe(), PipelineConfig())
    return TestClient(_server_module().create_app(str(models), rvc))


def _wav(seconds=1.5, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1), sr, format="WAV")
    return buf.getvalue()


def test_index_and_voices(client):
    assert "RVC" in client.get("/").text
    voices = client.get("/api/voices").json()
    assert voices == [{"id": "alice", "name": "Alice", "version": "v2", "sample_rate": 16000, "pitch": True, "index": True, "blend": None}]


def test_convert_round_trip(client):
    r = client.post("/api/convert", files={"audio": ("in.wav", _wav(), "audio/wav")}, data={"voice": "alice", "pitch": "3"})
    assert r.status_code == 200, r.text
    audio, sr = sf.read(io.BytesIO(r.content))
    assert sr == 16000 and abs(len(audio) / sr - 1.5) < 0.05
    assert float(r.headers["X-Duration"]) > 1.4


def test_convert_errors(client):
    bad = client.post("/api/convert", files={"audio": ("x.wav", b"not audio", "audio/wav")}, data={"voice": "alice"})
    assert bad.status_code == 400
    short = client.post("/api/convert", files={"audio": ("x.wav", _wav(0.1), "audio/wav")}, data={"voice": "alice"})
    assert short.status_code == 400
    missing = client.post("/api/convert", files={"audio": ("x.wav", _wav(), "audio/wav")}, data={"voice": "nobody"})
    assert missing.status_code == 404
