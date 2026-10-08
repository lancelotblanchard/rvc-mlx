"""
HuBERT / ContentVec: MLX forward vs the fairseq-layout PyTorch reference, through the real conversion path.

A small random `TorchHubert` (same topology as `hubert_base.pt`, narrower) is saved the way RVC ships it — a fairseq
checkpoint dict with a non-tensor config object — converted with `convert_hubert_checkpoint`, loaded back with
`HubertModel.from_pretrained`, and compared layer-for-layer. A second test re-saves the same weights under Hugging Face
`transformers` key names to cover ContentVec exports.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
import pytest
import torch

from rvc_mlx.convert import convert_hubert_checkpoint
from rvc_mlx.hubert import HubertConfig, HubertModel
from rvc_mlx.io import read_metadata
from tests.reference.hubert_torch import TorchHubert

SMALL = dict(embed_dim=48, ffn_dim=64, num_heads=4, num_layers=3, conv_pos=16, conv_pos_groups=4, final_dim=24)
SMALL_CONV = ((32, 10, 5),) + ((32, 3, 2),) * 4 + ((32, 2, 2),) * 2


@pytest.fixture(scope="module")
def torch_hubert():
    torch.manual_seed(0)
    model = TorchHubert(conv_layers=SMALL_CONV, **SMALL).eval()
    with torch.no_grad():
        # weight_norm initialises g = ||v||, which would make folding a no-op; perturb it so the test can tell.
        g = model.encoder.pos_conv[0].weight_g
        g.mul_(torch.rand_like(g) + 0.5)
        for m in model.modules():
            if isinstance(m, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
                m.weight.normal_(1.0, 0.2)
                m.bias.normal_(0.0, 0.2)
    return model


@pytest.fixture(scope="module")
def audio():
    rng = np.random.default_rng(0)
    return (0.3 * rng.standard_normal((1, 8000))).astype(np.float32)


def _torch_features(model, audio, layer, proj=False):
    with torch.no_grad():
        x, _ = model.extract_features(torch.from_numpy(audio), output_layer=layer)
        if proj:
            x = model.final_proj(x)
    return x.numpy()


class FakeDictConfig:
    """Stands in for omegaconf's DictConfig: a class the converter must not try to import when unpickling."""

    def __init__(self, content):
        self._content = content


def _unimportable(module, name):
    # A class whose pickled reference points at a module that doesn't exist, so a normal unpickler would fail.
    import sys

    cls = type(name, (FakeDictConfig,), {})
    cls.__module__ = module
    cls.__qualname__ = name
    mod = type(sys)(module)
    setattr(mod, name, cls)
    sys.modules[module] = mod  # registered only while saving; removed again in _fairseq_checkpoint
    return cls


def _fairseq_checkpoint(model):
    # `args` is an argparse.Namespace and `cfg` an omegaconf object in real fairseq checkpoints. Both force the
    # non-weights_only loading path; the unknown class must be stubbed out rather than imported.
    cfg_cls = _unimportable("tests.test_hubert_unimportable_module", "DictConfig")
    return {
        "args": argparse.Namespace(encoder_attention_heads=SMALL["num_heads"], arch="hubert"),
        "cfg": cfg_cls({"model": {"encoder_attention_heads": SMALL["num_heads"]}}),
        "model": model.state_dict(),
    }


def _save_fairseq(model, path):
    import sys

    torch.save(_fairseq_checkpoint(model), path)
    sys.modules.pop("tests.test_hubert_unimportable_module", None)
    with pytest.raises(Exception):  # a plain unpickler can no longer resolve the config class
        torch.load(path, weights_only=False)


_HF_RENAMES = [
    ("feature_extractor.conv_layers.0.2.", "feature_extractor.conv_layers.0.layer_norm."),
    ("layer_norm.", "feature_projection.layer_norm."),
    ("post_extract_proj.", "feature_projection.projection."),
    ("encoder.pos_conv.0.", "encoder.pos_conv_embed.conv."),
]


def _to_hf_names(state):
    out = {}
    for k, v in state.items():
        k2 = k
        if k.startswith("feature_extractor.conv_layers.") and ".0.weight" in k:
            k2 = k.replace(".0.weight", ".conv.weight")
        for a, b in _HF_RENAMES:
            if k2.startswith(a):
                k2 = b + k2[len(a):]
                break
        k2 = (
            k2.replace(".self_attn_layer_norm.", ".layer_norm.")
            .replace(".self_attn.", ".attention.")
            .replace(".fc1.", ".feed_forward.intermediate_dense.")
            .replace(".fc2.", ".feed_forward.output_dense.")
        )
        out[k2] = v
    return out


class TestHubertConversion:
    def test_fairseq_checkpoint_round_trip(self, tmp_path, torch_hubert, audio):
        pt = tmp_path / "hubert_base.pt"
        _save_fairseq(torch_hubert, pt)
        out = convert_hubert_checkpoint(str(pt), dtype="float32")
        assert out == str(tmp_path / "hubert_base.safetensors")

        meta = read_metadata(out)
        assert meta["kind"] == "hubert"
        cfg = HubertConfig.from_json(meta["config"])
        assert cfg.num_heads == SMALL["num_heads"] and cfg.num_layers == SMALL["num_layers"]
        assert cfg.conv_pos == SMALL["conv_pos"] and cfg.conv_pos_groups == SMALL["conv_pos_groups"]

        model = HubertModel.from_pretrained(out)
        for layer in (1, 2, 3):
            ref = _torch_features(torch_hubert, audio, layer)
            got = np.array(model.extract_features(mx.array(audio), output_layer=layer))
            np.testing.assert_allclose(got, ref, atol=2e-4, rtol=1e-3)

    def test_rvc_v1_and_v2_features(self, tmp_path, torch_hubert, audio):
        pt = tmp_path / "hubert_base.pt"
        _save_fairseq(torch_hubert, pt)
        model = HubertModel.from_pretrained(convert_hubert_checkpoint(str(pt), dtype="float32"))
        # Our small model has 3 layers, so "v2" (layer 12) runs all of them; v1 reads layer 9 which also clamps to 3.
        v2 = np.array(model(mx.array(audio), version="v2"))
        np.testing.assert_allclose(v2, _torch_features(torch_hubert, audio, 3), atol=2e-4, rtol=1e-3)
        v1 = np.array(model(mx.array(audio), version="v1"))
        np.testing.assert_allclose(v1, _torch_features(torch_hubert, audio, 9, proj=True), atol=2e-4, rtol=1e-3)
        assert v1.shape[-1] == SMALL["final_dim"]
        with pytest.raises(ValueError):
            model(mx.array(audio), version="v3")

    def test_huggingface_names(self, tmp_path, torch_hubert, audio):
        state = _to_hf_names(torch_hubert.state_dict())
        assert "feature_projection.projection.weight" in state
        pt = tmp_path / "pytorch_model.bin"
        torch.save(state, pt)
        model = HubertModel.from_pretrained(convert_hubert_checkpoint(str(pt), dtype="float32"))
        # HF checkpoints carry no config, so the head count falls back to HuBERT-Base's 12; rebuild with ours.
        model_cfg = model.cfg
        model_cfg.num_heads = SMALL["num_heads"]
        fixed = HubertModel(model_cfg)
        fixed.update(model.parameters())
        ref = _torch_features(torch_hubert, audio, 3)
        got = np.array(fixed.extract_features(mx.array(audio), output_layer=3))
        np.testing.assert_allclose(got, ref, atol=2e-4, rtol=1e-3)

    def test_float16_storage(self, tmp_path, torch_hubert, audio):
        pt = tmp_path / "hubert_base.pt"
        _save_fairseq(torch_hubert, pt)
        out = convert_hubert_checkpoint(str(pt), out_path=str(tmp_path / "h16.safetensors"))  # default float16
        stored = mx.load(out)
        assert all(v.dtype == mx.float16 for v in stored.values())
        got = np.array(HubertModel.from_pretrained(out).extract_features(mx.array(audio), output_layer=3))
        ref = _torch_features(torch_hubert, audio, 3)
        assert np.abs(got - ref).max() < 0.05 * np.abs(ref).max()


def test_default_config_matches_hubert_base():
    cfg = HubertConfig()
    assert [c[0] for c in cfg.conv_layers] == [512] * 7
    assert np.prod([c[2] for c in cfg.conv_layers]) == 320  # 50 frames / s at 16 kHz
    assert HubertConfig.from_json(cfg.to_json()) == cfg
