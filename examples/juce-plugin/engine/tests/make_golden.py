"""
Write small models and golden outputs from the Python implementation for the C++ parity test.

    python examples/juce-plugin/engine/tests/make_golden.py /tmp/rvc-golden
    ./build/rvc_parity_test /tmp/rvc-golden

With `--full <models_dir>` it also records golden outputs for full-size converted models (slow on CPU).
Run from the repository root (needs the dev extras: torch, faiss-cpu).
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

if sys.platform.startswith("linux"):
    mx.disable_compile()  # see tests/conftest.py

from rvc_mlx import _torch_ref  # noqa: E402
from rvc_mlx import rmvpe as rmvpe_mod  # noqa: E402
from rvc_mlx.convert import convert_hubert_checkpoint, convert_rmvpe_checkpoint, convert_voice_checkpoint  # noqa: E402
from rvc_mlx.dsp import change_rms, highpass  # noqa: E402
from rvc_mlx.hubert import HubertModel  # noqa: E402
from rvc_mlx.index import retrieve_blended  # noqa: E402
from rvc_mlx.pipeline import Pipeline, PipelineConfig  # noqa: E402
from rvc_mlx.rmvpe import E2E, RMVPE  # noqa: E402
from rvc_mlx.synthesizer import sine_excitation  # noqa: E402
from rvc_mlx.voice import Voice, blend_weights  # noqa: E402
from tests.reference.hubert_torch import TorchHubert  # noqa: E402
from tests.reference.synth_torch import Synthesizer as TorchSynthesizer  # noqa: E402

SMALL_E2E = dict(n_blocks=1, n_gru=1, kernel_size=(2, 2), en_de_layers=2, inter_layers=1, in_channels=1, en_out_channels=4)
CHUNKING = dict(x_pad=1, x_query=1, x_center=3, x_max=4)


def signal(seconds, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * (180 + 40 * np.sin(2 * np.pi * 0.5 * t)) * t) + 0.05 * rng.standard_normal(t.size)
    x *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t) ** 2
    x[int(0.37 * len(x)) : int(0.45 * len(x))] *= 0.01
    return x.astype(np.float32)


def voice_config(resblock="1"):
    rk, rd = ([3, 5], [[1, 3, 5], [1, 3, 5]]) if resblock == "1" else ([3, 5], [[1, 3], [1, 3]])
    return [33, 32, 16, 16, 32, 2, 2, 3, 0, resblock, rk, rd, [10, 4, 4], 32, [20, 8, 8], 3, 8, 16000]


def small_models(out):
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    torch.manual_seed(0)
    e2e = _torch_ref.TorchE2E(**SMALL_E2E)
    from rvc_mlx._convert_bridge import randomize_bn_stats

    randomize_bn_stats(e2e)
    torch.save(e2e.state_dict(), f"{raw}/rmvpe.pt")
    saved = rmvpe_mod._DEFAULT_E2E_CONFIG, _torch_ref.DEFAULT_E2E_CONFIG
    rmvpe_mod._DEFAULT_E2E_CONFIG = _torch_ref.DEFAULT_E2E_CONFIG = SMALL_E2E  # convert the small architecture
    try:
        convert_rmvpe_checkpoint(f"{raw}/rmvpe.pt", f"{out}/rmvpe.safetensors", dtype="float32")
    finally:
        rmvpe_mod._DEFAULT_E2E_CONFIG, _torch_ref.DEFAULT_E2E_CONFIG = saved

    conv = ((32, 10, 5),) + ((32, 3, 2),) * 4 + ((32, 2, 2),) * 2
    th = TorchHubert(conv_layers=conv, embed_dim=768, ffn_dim=64, num_heads=12, num_layers=2, conv_pos=16, conv_pos_groups=16)
    torch.save({"model": th.state_dict()}, f"{raw}/hubert_base.pt")
    convert_hubert_checkpoint(f"{raw}/hubert_base.pt", f"{out}/hubert.safetensors", dtype="float32")

    rng = np.random.default_rng(0)
    specs = {"v2_f0": ("v2", True, "1"), "v1_f0": ("v1", True, "1"), "v2_nof0": ("v2", False, "2"), "v2_f0_b": ("v2", True, "1")}
    for i, (name, (version, f0, rb)) in enumerate(specs.items()):
        torch.manual_seed(100 + i)
        cfg = voice_config(rb)
        m = TorchSynthesizer(*cfg, version=version, f0=f0)
        torch.save({"weight": m.state_dict(), "config": cfg, "f0": int(f0), "version": version}, f"{raw}/{name}.pth")
        np.save(f"{raw}/{name}.npy", rng.standard_normal((300, 768 if version == "v2" else 256)).astype(np.float32))
        convert_voice_checkpoint(f"{raw}/{name}.pth", f"{out}/{name}.safetensors", index_path=f"{raw}/{name}.npy",
                                 name=name, dtype="float32")


def golden_small(out):
    g = {}
    rmvpe_w = {k: v for k, v in mx.load(f"{out}/rmvpe.safetensors").items() if k != "mel_basis"}
    e2e = E2E(**SMALL_E2E)
    e2e.load_weights(list(rmvpe_w.items()))
    e2e.eval()
    rmvpe = RMVPE(e2e)
    hubert = HubertModel.from_pretrained(f"{out}/hubert.safetensors")

    audio = signal(7.0)
    g["audio"] = audio
    short = audio[: 16000 * 2]
    g["hubert_v2"] = hubert(mx.array(short)[None], version="v2")
    g["hubert_v1"] = hubert(mx.array(short)[None], version="v1")
    mel = rmvpe.mel_extractor(mx.array(short)[None])
    g["rmvpe_mel"] = mel
    g["rmvpe_salience"] = rmvpe.mel2hidden(mel)
    g["rmvpe_f0"] = mx.array(rmvpe.infer_from_audio(short).astype(np.float32))

    rng = np.random.default_rng(1)
    f0 = np.where(rng.random(50) < 0.8, rng.uniform(80, 600, 50), 0).astype(np.float32)[None]
    noise = rng.standard_normal((1, 50 * 160, 1)).astype(np.float32)
    g["sine_f0"], g["sine_noise"] = mx.array(f0), mx.array(noise)
    g["sine_out"] = sine_excitation(mx.array(f0), 160, 16000, noise=mx.array(noise))

    T = 40
    for name in ("v2_f0", "v1_f0", "v2_nof0"):
        v = Voice.load(f"{out}/{name}.safetensors")
        phone = rng.standard_normal((1, T, v.cfg.feature_dim)).astype(np.float32)
        pitch_hz = np.where(rng.random(T) < 0.8, rng.uniform(80, 600, T), 0).astype(np.float32)[None]
        from rvc_mlx.pipeline import coarse_f0

        coarse = coarse_f0(pitch_hz[0].astype(np.float64).copy())[None]
        prior = rng.standard_normal((1, T, v.cfg.inter_channels)).astype(np.float32)
        nsf = rng.standard_normal((1, T * v.cfg.hop_length, 1)).astype(np.float32)
        y = v.model.infer(mx.array(phone), mx.array([T]), mx.array(coarse), mx.array(pitch_hz), mx.array([2]),
                          prior_noise=mx.array(prior), nsf_noise=mx.array(nsf))
        g.update({f"synth_{name}_{k}": mx.array(a) for k, a in
                  dict(phone=phone, pitch=coarse.astype(np.int32), pitchf=pitch_hz, prior=prior, nsf=nsf).items()})
        g[f"synth_{name}_out"] = y

        pipe = Pipeline(v.sample_rate, PipelineConfig(**CHUNKING), rmvpe=rmvpe)
        assert len(pipe.split_points(highpass(audio.astype(np.float64)))) == 2
        for tag, kw in {"a": dict(f0_up_key=3, index_rate=0.75, protect=0.33, rms_mix_rate=0.25),
                        "b": dict(f0_up_key=-5, index_rate=0.0, protect=0.5, rms_mix_rate=1.0)}.items():
            g[f"pipe_{name}_{tag}"] = mx.array(pipe.pipeline(hubert, v, 1, audio, deterministic=True, **kw))

    # Edge case: the last split point lands in the final frame, so the chunk slice must clamp at the end of the
    # padded audio (numpy slicing does; the C++/Swift ports have to do it explicitly).
    edge = (0.3 * np.random.default_rng(5).standard_normal(96050)).astype(np.float32)  # no quiet stretch except...
    edge[-50:] = 0.0  # ...the tail, so the 6 s split point lands within the last frame
    v = Voice.load(f"{out}/v2_f0.safetensors")
    pipe = Pipeline(v.sample_rate, PipelineConfig(**CHUNKING), rmvpe=rmvpe)
    ts = pipe.split_points(highpass(edge.astype(np.float64)))
    assert ts and ts[-1] // 160 * 160 > len(edge) - 160, (ts, len(edge))
    g["edge_audio"] = edge
    g["pipe_edge"] = mx.array(pipe.pipeline(hubert, v, 0, edge, deterministic=True, f0_up_key=0, index_rate=0.5))

    a, b = Voice.load(f"{out}/v2_f0.safetensors"), Voice.load(f"{out}/v2_f0_b.safetensors")
    blended = blend_weights([a, b], [0.3, 0.7])
    for key in ("dec.ups.0.weight", "enc_p.emb_phone.weight", "emb_g.weight", "flow.flows.2.enc.cond_layer.weight"):
        g[f"blend_{key}"] = blended[key]
    q = mx.array(rng.standard_normal((1, 25, 768)).astype(np.float32))
    g["retrieve_q"] = q
    g["retrieve_out"] = retrieve_blended(q, [a.index, b.index], [0.3, 0.7])

    x = signal(1.0, seed=3).astype(np.float64)
    g["dsp_x"] = mx.array(x.astype(np.float32))
    g["dsp_highpass"] = mx.array(highpass(x.astype(np.float32).astype(np.float64)).astype(np.float32))
    tgt = rng.standard_normal(40000).astype(np.float32)
    g["dsp_rms_target"] = mx.array(tgt)
    g["dsp_rms_out"] = mx.array(change_rms(x.astype(np.float32).astype(np.float64), 16000, tgt.astype(np.float64), 40000, 0.25).astype(np.float32))
    arrays = {k: mx.array(v) for k, v in g.items()}
    arrays = {k: v if v.dtype == mx.int32 else v.astype(mx.float32) for k, v in arrays.items()}
    mx.save_safetensors(f"{out}/golden.safetensors", arrays)


def golden_full(models, out):
    from rvc_mlx import RVC

    rvc = RVC.from_pretrained(models)
    v = Voice.load(sorted(p for p in (os.path.join(models, "voices", f) for f in os.listdir(os.path.join(models, "voices"))) if p.endswith(".safetensors"))[0])
    audio = signal(1.5, seed=7)
    pipe = Pipeline(v.sample_rate, PipelineConfig(), rmvpe=rvc.rmvpe)
    g = {
        "audio": audio,
        "hubert_v2": rvc.hubert(mx.array(audio)[None], version="v2"),
        "rmvpe_f0": mx.array(rvc.rmvpe.infer_from_audio(audio).astype(np.float32)),
        "pipe": mx.array(pipe.pipeline(rvc.hubert, v, 0, audio, f0_up_key=2, deterministic=True)),
    }
    mx.save_safetensors(f"{out}/golden_full.safetensors", {k: mx.array(v).astype(mx.float32) for k, v in g.items()},
                        metadata={"voice": v.metadata.get("name", "")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--full", help="a converted models/ folder (hubert, rmvpe, voices/) to record full-size goldens")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    small_models(args.out)
    golden_small(args.out)
    if args.full:
        golden_full(args.full, args.out)
    print("golden data written to", args.out)
