"""
RVC synthesizer: MLX vs the transcribed PyTorch reference, through the real voice-conversion path.

Each case builds a small random `Synthesizer` in torch, saves it in RVC's export format (`{"weight", "config",
"version", "f0", ...}`), converts it with `convert_voice_checkpoint`, loads it with `Voice.load`, and compares outputs
with identical injected noise. Shapes are kept small but every code path (relative attention with T both shorter and
longer than the attention window, padding masks, flows, NSF excitation, both ResBlock types, v1/v2, with/without f0)
is exercised.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

from rvc_mlx.convert import convert_voice_checkpoint
from rvc_mlx.synthesizer import SynthConfig, sine_excitation
from rvc_mlx.voice import Voice
from tests.reference.synth_torch import SineGen, Synthesizer as TorchSynthesizer


def _rvc_config(resblock="1", sr=16000, ups=(4, 2, 2), upk=(8, 4, 4), spk=3):
    rk = [3, 5]
    rd = [[1, 3, 5], [1, 3, 5]] if resblock == "1" else [[1, 3], [1, 3]]
    # spec_channels, segment_size, inter, hidden, filter, heads, layers, kernel, p_dropout, resblock, rk, rd,
    # upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr
    return [33, 32, 16, 16, 32, 2, 2, 3, 0, resblock, rk, rd, list(ups), 32, list(upk), spk, 8, sr]


def _build_torch(config, version, f0, seed=0):
    torch.manual_seed(seed)
    model = TorchSynthesizer(*config, version=version, f0=f0).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("weight_g"):
                p.mul_(torch.rand_like(p) + 0.5)  # make weight-norm folding observable
            if name.startswith("flow.") and ".post." in name:
                p.normal_(0, 0.1)
    return model


def _export(model, config, version, f0, path, half=False):
    weight = {k: (v.half() if half else v) for k, v in model.state_dict().items()}
    ckpt = {"weight": weight, "config": config, "info": "unit-test", "sr": "16k", "f0": int(f0), "version": version}
    torch.save(ckpt, path)


def _inputs(cfg_dim, T, f0, seed=0):
    rng = np.random.default_rng(seed)
    phone = rng.standard_normal((1, T, cfg_dim)).astype(np.float32)
    f0_hz = np.where(rng.random(T) < 0.8, rng.uniform(80, 500, T), 0.0).astype(np.float32)
    f0_mel = 1127 * np.log(1 + f0_hz / 700)
    lo, hi = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    coarse = np.where(f0_mel > 0, (f0_mel - lo) * 254 / (hi - lo) + 1, 1)
    coarse = np.rint(np.clip(coarse, 1, 255)).astype(np.int64)[None]
    return phone, coarse, f0_hz[None]


def _compare(tmp_path, version, f0, resblock, T, half=False, atol=2e-4):
    config = _rvc_config(resblock)
    tmodel = _build_torch(config, version, f0)
    pth = tmp_path / "voice.pth"
    _export(tmodel, config, version, f0, pth, half=half)
    voice = Voice.load(convert_voice_checkpoint(str(pth), dtype="float32"))
    assert voice.version == version and voice.has_f0 == f0 and voice.sample_rate == 16000

    cfg = voice.cfg
    phone, coarse, f0_hz = _inputs(cfg.feature_dim, T, f0)
    rng = np.random.default_rng(1)
    prior = rng.standard_normal((1, T, cfg.inter_channels)).astype(np.float32)
    nsf = rng.standard_normal((1, T * cfg.hop_length, 1)).astype(np.float32)

    if half:  # compare against the same fp16-rounded weights torch would see after loading the export
        tmodel = tmodel.half().float()
    with torch.no_grad():
        ref, _, _ = tmodel.infer(
            torch.from_numpy(phone),
            torch.tensor([T]),
            torch.from_numpy(coarse) if f0 else None,
            torch.from_numpy(f0_hz) if f0 else None,
            torch.tensor([1]),
            prior_noise=torch.from_numpy(prior).transpose(1, 2),
            nsf_noise=torch.from_numpy(nsf),
        )
    got = voice.model.infer(
        mx.array(phone),
        mx.array([T]),
        mx.array(coarse) if f0 else None,
        mx.array(f0_hz) if f0 else None,
        mx.array([1]),
        prior_noise=mx.array(prior),
        nsf_noise=mx.array(nsf),
    )
    ref = ref.numpy()[:, 0]
    got = np.array(got)
    assert got.shape == ref.shape == (1, T * cfg.hop_length)
    np.testing.assert_allclose(got, ref, atol=atol, rtol=1e-3)


@pytest.mark.parametrize("T", [7, 30])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_nsf_synthesizer_matches_torch(tmp_path, version, T):
    _compare(tmp_path, version, f0=True, resblock="1", T=T)


@pytest.mark.parametrize("resblock", ["1", "2"])
def test_nono_synthesizer_matches_torch(tmp_path, resblock):
    _compare(tmp_path, "v2", f0=False, resblock=resblock, T=25)


def test_half_precision_export(tmp_path):
    # Real exports are fp16. Conversion upcasts before folding weight norm, so results track the fp16 weights closely.
    _compare(tmp_path, "v2", f0=True, resblock="1", T=20, half=True, atol=2e-3)


def test_text_encoder_padding_mask(tmp_path):
    config = _rvc_config()
    tmodel = _build_torch(config, "v2", True)
    pth = tmp_path / "voice.pth"
    _export(tmodel, config, "v2", True, pth)
    voice = Voice.load(convert_voice_checkpoint(str(pth), dtype="float32"))
    T, valid = 24, 17
    phone, coarse, _ = _inputs(768, T, True)
    with torch.no_grad():
        m, logs, mask = tmodel.enc_p(torch.from_numpy(phone), torch.from_numpy(coarse), torch.tensor([valid]))
    gm, glogs, gmask = voice.model.enc_p(mx.array(phone), mx.array(coarse), mx.array([valid]))
    np.testing.assert_allclose(np.array(gm), m.numpy().transpose(0, 2, 1), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(np.array(glogs), logs.numpy().transpose(0, 2, 1), atol=1e-4, rtol=1e-3)
    np.testing.assert_array_equal(np.array(gmask)[..., 0], mask.numpy()[:, 0])


def test_sine_excitation_long_input_matches_torch():
    # 40k-voice geometry: 400 samples per frame, 6 s of frames. Phase must stay accurate across 240k samples.
    rng = np.random.default_rng(0)
    T, upp, sr = 600, 400, 40000
    f0 = np.where(rng.random(T) < 0.7, rng.uniform(60, 900, T), 0.0).astype(np.float32)[None]
    noise = rng.standard_normal((1, T * upp, 1)).astype(np.float32)
    gen = SineGen(sr)
    gen.noise = torch.from_numpy(noise)
    ref, _, _ = gen(torch.from_numpy(f0), upp)
    got = sine_excitation(mx.array(f0), upp, sr, noise=mx.array(noise))
    np.testing.assert_allclose(np.array(got), ref.numpy(), atol=2e-3)


def test_config_round_trip_and_rvc_layout():
    cfg = SynthConfig.from_rvc(
        [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [12, 10, 2, 2], 512,
         [24, 20, 4, 4], 109, 256, 48000],
        version="v2",
    )
    assert cfg.hop_length == 480 and cfg.feature_dim == 768 and cfg.sr == 48000
    assert SynthConfig.from_json(cfg.to_json()) == cfg
    with pytest.raises(ValueError):
        SynthConfig(version="v3")


def test_rejects_training_checkpoints(tmp_path):
    p = tmp_path / "G_1000.pth"
    torch.save({"model": {}, "iteration": 1000}, p)
    with pytest.raises(ValueError, match="RVC voice export"):
        convert_voice_checkpoint(str(p))
