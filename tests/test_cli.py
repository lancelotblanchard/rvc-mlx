"""The `rvc-mlx` CLI, end to end on tiny models: convert-base -> convert-voice -> info -> blend -> infer."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from rvc_mlx import _torch_ref
from rvc_mlx import rmvpe as rmvpe_mod
from rvc_mlx.cli import main
from rvc_mlx.voice import Voice
from tests.reference.hubert_torch import TorchHubert
from tests.reference.synth_torch import Synthesizer as TorchSynthesizer

SMALL_E2E = dict(n_blocks=1, n_gru=1, kernel_size=(2, 2), en_de_layers=2, inter_layers=1, in_channels=1, en_out_channels=4)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(rmvpe_mod, "_DEFAULT_E2E_CONFIG", SMALL_E2E)
    monkeypatch.setattr(_torch_ref, "DEFAULT_E2E_CONFIG", SMALL_E2E)
    torch.manual_seed(0)
    torch.save(_torch_ref.TorchE2E(**SMALL_E2E).state_dict(), tmp_path / "rmvpe.pt")
    conv = ((32, 10, 5),) + ((32, 3, 2),) * 4 + ((32, 2, 2),) * 2
    th = TorchHubert(conv_layers=conv, embed_dim=768, ffn_dim=64, num_heads=12, num_layers=2, conv_pos=16, conv_pos_groups=16)
    torch.save({"args": argparse.Namespace(encoder_attention_heads=12), "model": th.state_dict()}, tmp_path / "hubert_base.pt")
    config = [33, 32, 16, 16, 32, 2, 2, 3, 0, "1", [3], [[1, 3, 5]], [10, 4, 4], 16, [20, 8, 8], 2, 8, 16000]
    for i, name in enumerate(("alice", "bob")):
        torch.manual_seed(10 + i)
        ts = TorchSynthesizer(*config, version="v2", f0=True)
        sd = {k: v.half() for k, v in ts.state_dict().items()}
        torch.save({"weight": sd, "config": config, "f0": 1, "version": "v2", "info": "epoch 1"}, tmp_path / f"{name}.pth")
    np.save(tmp_path / "alice_fea.npy", np.random.default_rng(0).standard_normal((64, 768)).astype(np.float32))
    t = np.arange(32000) / 16000
    sf.write(tmp_path / "in.wav", (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 16000)
    return tmp_path


def test_cli_end_to_end(workspace, capsys):
    w = workspace
    models = w / "models"
    main(["convert-base", "--hubert", str(w / "hubert_base.pt"), "--rmvpe", str(w / "rmvpe.pt"), "-o", str(models)])
    assert (models / "hubert.safetensors").exists() and (models / "rmvpe.safetensors").exists()

    main(["convert-voice", str(w / "alice.pth"), "--index", str(w / "alice_fea.npy"), "-o", str(models / "voices"), "--name", "Alice"])
    main(["convert-voice", str(w / "bob.pth"), "-o", str(models / "voices" / "bob.safetensors")])
    alice = Voice.load(str(models / "voices" / "alice.safetensors"))
    assert alice.name == "Alice" and alice.index.shape == (64, 768) and alice.info == "epoch 1"

    capsys.readouterr()
    main(["info", str(models / "voices" / "alice.safetensors")])
    meta = json.loads(capsys.readouterr().out)
    assert meta["kind"] == "voice" and meta["config"]["sr"] == 16000 and meta["index_size"] == "64"

    mix = models / "voices" / "mix.safetensors"
    main(["blend", f"{models / 'voices' / 'alice.safetensors'}:0.25", str(models / "voices" / "bob.safetensors"), "-o", str(mix)])
    assert "Alice:0.200, bob:0.800" in Voice.load(str(mix)).metadata["merged_from"]  # weights 0.25 and 1 -> 0.2 / 0.8

    out = w / "out.wav"
    main(["infer", str(w / "in.wav"), str(out), "-m", str(models), "-v", str(mix), "--pitch", "2", "--sample-rate", "22050"])
    audio, sr = sf.read(out)
    assert sr == 22050 and abs(len(audio) / sr - 2.0) < 0.05 and np.isfinite(audio).all()


def test_convert_base_requires_an_input(capsys):
    with pytest.raises(SystemExit):
        main(["convert-base"])
