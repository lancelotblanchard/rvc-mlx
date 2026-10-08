"""
End-to-end conversion: `rvc_mlx.pipeline.Pipeline.pipeline` vs a transcription of RVC's own pipeline
(`tests/reference/pipeline_ref.py`) running on the PyTorch reference models, plus the retrieval / voice / API layers.

Small random models stand in for real checkpoints; the input is long enough (7 s with a 4 s `x_max`) to exercise the
silence-seeking chunker. f0 comes from a deterministic stub shared by both sides (RMVPE is tested on its own), and the
models' sampling noise is disabled.
"""

from __future__ import annotations

import argparse

import faiss
import librosa
import mlx.core as mx
import numpy as np
import pytest
import torch

from rvc_mlx.api import RVC
from rvc_mlx.convert import convert_hubert_checkpoint, convert_voice_checkpoint, read_index_vectors
from rvc_mlx.dsp import change_rms, frame_rms, highpass, interp_linear
from rvc_mlx.hubert import HubertModel
from rvc_mlx.index import kmeans, knn, retrieve, retrieve_blended
from rvc_mlx.pipeline import Pipeline, PipelineConfig, coarse_f0
from rvc_mlx.rmvpe import E2E, RMVPE
from rvc_mlx.voice import Voice, blend_voices, check_compatible
from tests.reference.hubert_torch import TorchHubert
from tests.reference.pipeline_ref import RefPipeline, build_faiss_index
from tests.reference.synth_torch import Synthesizer as TorchSynthesizer

SR = 16000
CHUNKING = dict(x_pad=1, x_query=1, x_center=3, x_max=4)


class StubRMVPE:
    """Deterministic f0 (100 frames/s, RMVPE's frame count) with voiced and unvoiced stretches."""

    def infer_from_audio(self, x, thred=0.03):
        n = 1 + len(x) // 160
        t = np.arange(n)
        f0 = 180 + 60 * np.sin(t / 37.0) + 20 * np.sin(t / 5.0)
        f0[(t // 45) % 4 == 3] = 0.0
        return f0.astype(np.float64)


def _signal(seconds=7.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 660 * t + 1) + 0.05 * rng.standard_normal(t.size)
    x *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t) ** 2
    x[int(2.6 * SR) : int(3.1 * SR)] *= 0.01  # a near-silent stretch for the chunker to find
    return x.astype(np.float32)


def _voice_config(sr=SR):
    # hop = 10 * 4 * 4 = 160 samples per 100 Hz frame at 16 kHz
    return [33, 32, 16, 16, 32, 2, 2, 3, 0, "1", [3, 5], [[1, 3, 5], [1, 3, 5]], [10, 4, 4], 32, [20, 8, 8], 2, 8, sr]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    torch.manual_seed(0)
    conv = ((32, 10, 5),) + ((32, 3, 2),) * 4 + ((32, 2, 2),) * 2
    th = TorchHubert(conv_layers=conv, embed_dim=768, ffn_dim=64, num_heads=12, num_layers=2, conv_pos=16,
                     conv_pos_groups=16, final_dim=256).eval()
    torch.save({"args": argparse.Namespace(encoder_attention_heads=12), "model": th.state_dict()}, tmp / "hubert_base.pt")
    hubert = HubertModel.from_pretrained(convert_hubert_checkpoint(str(tmp / "hubert_base.pt"), dtype="float32"))

    out = {"torch_hubert": th, "hubert": hubert, "dir": tmp}
    rng = np.random.default_rng(0)
    for version, f0 in (("v2", True), ("v1", True), ("v2", False)):
        torch.manual_seed(1)
        config = _voice_config()
        ts = TorchSynthesizer(*config, version=version, f0=f0).eval()
        name = f"{version}_{'f0' if f0 else 'nof0'}"
        torch.save({"weight": ts.state_dict(), "config": config, "f0": int(f0), "version": version, "info": ""}, tmp / f"{name}.pth")
        bank = rng.standard_normal((400, 768 if version == "v2" else 256)).astype(np.float32)
        build_faiss_index(bank, str(tmp / f"{name}.index"))
        voice = Voice.load(convert_voice_checkpoint(str(tmp / f"{name}.pth"), index_path=str(tmp / f"{name}.index"), dtype="float32"))
        out[name] = (ts, voice, bank)
    return out


@pytest.mark.parametrize(
    "name,pitch,index_rate,protect,rms_mix",
    [
        ("v2_f0", 0, 0.75, 0.33, 0.25),
        ("v1_f0", 5, 0.5, 0.5, 1.0),
        ("v2_nof0", 0, 0.0, 0.33, 0.6),
    ],
)
def test_pipeline_matches_rvc_reference(models, name, pitch, index_rate, protect, rms_mix):
    ts, voice, bank = models[name]
    audio = _signal()

    ref_pipe = RefPipeline(SR, f0_fn=StubRMVPE().infer_from_audio, **CHUNKING)
    flat = faiss.IndexFlatL2(bank.shape[1])
    flat.add(bank)
    ref = ref_pipe.pipeline(
        models["torch_hubert"], ts, 0, audio.astype(np.float64), pitch, flat, bank, index_rate,
        1 if voice.has_f0 else 0, SR, rms_mix, voice.version, protect,
    )

    pipe = Pipeline(SR, PipelineConfig(**CHUNKING), rmvpe=StubRMVPE())
    assert len(pipe.split_points(highpass(audio.astype(np.float64)))) == 2  # the chunked path really runs
    got = pipe.pipeline(
        models["hubert"], voice, 0, audio, f0_up_key=pitch, index_rate=index_rate, rms_mix_rate=rms_mix,
        protect=protect, deterministic=True,
    )
    assert got.dtype == np.float32
    assert got.shape == ref.shape
    # Like upstream, each chunk comes out ~1 frame short (HuBERT framing); the total drift stays a few frames.
    assert abs(len(got) - len(audio)) <= 2 * 160 * (1 + len(pipe.split_points(highpass(audio.astype(np.float64)))))
    np.testing.assert_allclose(got, ref, atol=2e-4 * max(1.0, np.abs(ref).max()), rtol=1e-3)


def test_blended_retrieval(models):
    # Retrieval over several banks (voice blending) is the weighted mean of single-bank retrievals.
    _, voice, bank = models["v2_nof0"]
    feats = mx.array(np.random.default_rng(3).standard_normal((1, 20, 768)).astype(np.float32))
    b1, b2 = mx.array(bank[:200]), mx.array(bank[200:])
    expect = 0.25 * retrieve(feats, b1) + 0.75 * retrieve(feats, b2)
    np.testing.assert_allclose(np.array(retrieve_blended(feats, [b1, b2], [1, 3])), np.array(expect), atol=1e-6)
    np.testing.assert_allclose(np.array(retrieve_blended(feats, [b1, b2], [1, 0])), np.array(retrieve(feats, b1)), atol=1e-6)
    with pytest.raises(ValueError):
        retrieve_blended(feats, [b1, b2], [1])

    pipe = Pipeline(SR, PipelineConfig(**CHUNKING), rmvpe=StubRMVPE())
    a = pipe.vc(models["hubert"], voice, 0, _signal(1.0), None, None, index_rate=1.0, protect=0.5,
                deterministic=True, index_banks=[b1, b2], index_weights=[1, 3])
    assert a.shape == (98 * 160,) and np.isfinite(a).all()  # 1 s -> 49 HuBERT frames -> 98 f0-rate frames

    _, f0_voice, _ = models["v2_f0"]
    with pytest.raises(ValueError, match="pitch-conditioned"):
        pipe.vc(models["hubert"], f0_voice, 0, _signal(1.0), None, None, index_rate=0.0, protect=0.5)


# --------------------------------------------------------------------------------------------- retrieval / index


def test_knn_and_retrieve_match_faiss():
    rng = np.random.default_rng(0)
    bank = rng.standard_normal((1000, 64)).astype(np.float32)
    q = rng.standard_normal((300, 64)).astype(np.float32)
    flat = faiss.IndexFlatL2(64)
    flat.add(bank)
    score, ix = flat.search(q, 8)
    d, idx = knn(mx.array(q), mx.array(bank), k=8, chunk=128)
    order = np.argsort(np.array(d), axis=1)
    np.testing.assert_array_equal(np.take_along_axis(np.array(idx), order, 1), ix)
    np.testing.assert_allclose(np.take_along_axis(np.array(d), order, 1), score, rtol=1e-4, atol=1e-3)

    w = np.square(1 / score)
    w /= w.sum(axis=1, keepdims=True)
    expected = np.sum(bank[ix] * w[..., None], axis=1)
    np.testing.assert_allclose(np.array(retrieve(mx.array(q), mx.array(bank))), expected, rtol=1e-4, atol=1e-4)


def test_read_index_vectors_from_ivf(tmp_path):
    rng = np.random.default_rng(1)
    bank = rng.standard_normal((500, 32)).astype(np.float32)
    build_faiss_index(bank, str(tmp_path / "x.index"))
    np.testing.assert_allclose(read_index_vectors(str(tmp_path / "x.index")), bank)
    np.save(tmp_path / "total_fea.npy", bank)
    np.testing.assert_allclose(read_index_vectors(str(tmp_path / "total_fea.npy")), bank)


def test_kmeans_reduces_bank():
    rng = np.random.default_rng(2)
    centers = rng.standard_normal((5, 16)) * 10
    x = (centers[rng.integers(0, 5, 2000)] + 0.1 * rng.standard_normal((2000, 16))).astype(np.float32)
    c = kmeans(x, 5, n_iter=30, seed=3)
    assert c.shape == (5, 16) and c.dtype == np.float32

    def quantisation_error(cents):
        return np.linalg.norm(x[:, None] - cents[None], axis=-1).min(axis=1).mean()

    # Random-point init (what RVC's MiniBatchKMeans(init="random") does): Lloyd steps never make things worse than the
    # initial centroids, and the result is reproducible for a given seed.
    init = x[np.random.default_rng(3).choice(len(x), 5, replace=False)]
    assert quantisation_error(c) <= quantisation_error(init) + 1e-4
    np.testing.assert_array_equal(kmeans(x, 5, n_iter=30, seed=3), c)


def test_convert_voice_shrinks_index(models, tmp_path):
    p = convert_voice_checkpoint(str(models["dir"] / "v2_f0.pth"), out_path=str(tmp_path / "small.safetensors"),
                                 index_path=str(models["dir"] / "v2_f0.index"), max_index_vectors=50)
    v = Voice.load(p)
    assert v.index.shape == (50, 768)
    assert v.index.dtype == mx.float32  # loaded at compute precision
    assert mx.load(p)["index.vectors"].dtype == mx.float16  # stored at half precision by default


def test_convert_voice_rejects_mismatched_index(models, tmp_path):
    np.save(tmp_path / "bad.npy", np.zeros((10, 256), np.float32))
    with pytest.raises(ValueError, match="needs"):
        convert_voice_checkpoint(str(models["dir"] / "v2_f0.pth"), out_path=str(tmp_path / "x.safetensors"),
                                 index_path=str(tmp_path / "bad.npy"))


# --------------------------------------------------------------------------------------------- voices / blending


def test_voice_save_load_round_trip(models, tmp_path):
    _, voice, _ = models["v2_f0"]
    p = voice.save(str(tmp_path / "rt.safetensors"), dtype=mx.float32)
    back = Voice.load(p)
    assert back.name == voice.name and back.cfg == voice.cfg
    for k, v in voice.weights().items():
        np.testing.assert_array_equal(np.array(back.weights()[k]), np.array(v))
    np.testing.assert_array_equal(np.array(back.index), np.array(voice.index))
    assert "v2" in repr(back)


def test_blend_voices(models, tmp_path):
    _, a, _ = models["v2_f0"]
    torch.manual_seed(7)
    config = _voice_config()
    ts = TorchSynthesizer(*config, version="v2", f0=True).eval()
    torch.save({"weight": ts.state_dict(), "config": config, "f0": 1, "version": "v2"}, tmp_path / "b.pth")
    b = Voice.load(convert_voice_checkpoint(str(tmp_path / "b.pth"), dtype="float32"))

    only_a = blend_voices([a, b], [1.0, 0.0])
    for k, v in a.weights().items():
        np.testing.assert_allclose(np.array(only_a.weights()[k]), np.array(v), atol=1e-6)
    half = blend_voices([a, b], [2.0, 2.0], name="mix")
    k = "dec.conv_pre.weight"
    np.testing.assert_allclose(np.array(half.weights()[k]), (np.array(a.weights()[k]) + np.array(b.weights()[k])) / 2, atol=1e-6)
    assert half.name == "mix" and "0.500" in half.metadata["merged_from"]
    assert half.index.shape[0] == a.index.shape[0]  # b has no index; a's bank is kept

    p = half.save(str(tmp_path / "mix.safetensors"))
    assert "merged_from" in Voice.load(p).metadata

    _, v1, _ = models["v1_f0"]
    with pytest.raises(ValueError, match="different architectures"):
        check_compatible([a, v1])
    with pytest.raises(ValueError):
        blend_voices([a, b], [0.0, 0.0])


# --------------------------------------------------------------------------------------------------- dsp helpers


def test_dsp_helpers_match_reference_libraries():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(12345)
    np.testing.assert_allclose(frame_rms(y, 800, 400), librosa.feature.rms(y=y, frame_length=800, hop_length=400)[0], rtol=1e-6)
    x = rng.standard_normal(37)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x)[None, None], size=1000, mode="linear")[0, 0].numpy()
    np.testing.assert_allclose(interp_linear(x, 1000), ref, rtol=1e-6, atol=1e-9)
    out = change_rms(y[:16000], 16000, rng.standard_normal(40000), 40000, 0.25)
    assert out.shape == (40000,) and np.isfinite(out).all()


def test_coarse_f0_bins():
    f0 = np.array([0.0, 49.0, 50.0, 1100.0, 5000.0, 440.0])
    c = coarse_f0(f0.copy())
    assert c.tolist()[:5] == [1, 1, 1, 255, 255]
    assert 1 < c[-1] < 255


# ---------------------------------------------------------------------------------------------------- public API


def _small_rmvpe():
    m = E2E(n_blocks=1, n_gru=1, kernel_size=(2, 2), en_de_layers=2, inter_layers=1, in_channels=1, en_out_channels=4)
    m.eval()
    return RMVPE(m)


def test_rvc_convert_api(models):
    _, voice, _ = models["v2_f0"]
    rvc = RVC(models["hubert"], _small_rmvpe(), PipelineConfig(**CHUNKING))
    audio = _signal(2.0)
    stereo44 = np.stack([librosa.resample(audio, orig_sr=SR, target_sr=44100)] * 2, axis=1)
    out, sr = rvc.convert(stereo44, 44100, voice, pitch=3, output_sample_rate=48000)
    assert sr == 48000 and out.ndim == 1
    assert abs(len(out) / sr - 2.0) < 0.05
    assert np.abs(out).max() <= 0.99 + 1e-6

    calls = []
    out2, sr2 = rvc.convert(audio, SR, voice, deterministic=True, progress=calls.append)
    out3, _ = rvc.convert(audio, SR, voice, deterministic=True)
    assert sr2 == voice.sample_rate and calls[-1] == 1.0
    np.testing.assert_array_equal(out2, out3)
