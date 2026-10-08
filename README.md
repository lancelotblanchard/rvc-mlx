# RVC-MLX

[![Tests](https://github.com/lancelotblanchard/rvc-mlx/actions/workflows/tests.yml/badge.svg)](https://github.com/lancelotblanchard/rvc-mlx/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/github/lancelotblanchard/rvc-mlx/graph/badge.svg?token=78POKEWDJ3)](https://codecov.io/github/lancelotblanchard/rvc-mlx)

> [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) voice conversion on Apple silicon,
> with [MLX](https://github.com/ml-explore/mlx): no PyTorch at runtime, from Python, Swift (iOS) and C++ (plug-ins).

## Features

- 🚀 Complete RVC inference in MLX: HuBERT / ContentVec content features, RMVPE pitch, the v1 / v2 synthesizers
  (32k / 40k / 48k, with or without pitch guidance) and the WebUI's conversion pipeline (index retrieval, consonant
  protection, loudness mixing, long-input chunking).
- 📦 One-step conversion of your existing RVC models (`.pth` + `.index`) into single `.safetensors` files.
- 🎛️ Voice blending: interpolate the weights of compatible voices.
- ✅ Tested against PyTorch: every module is compared with a reference that keeps RVC's original module layout,
  and the full pipeline is checked against a transcription of the upstream one.

## Quick start

```bash
pip install -e ".[convert]"

rvc-mlx convert-base  --hubert hubert_base.pt --rmvpe rmvpe.pt -o models/
rvc-mlx convert-voice alice.pth --index added_IVF1024_Flat_nprobe_1_alice_v2.index -o models/voices/
rvc-mlx infer -m models/ -v models/voices/alice.safetensors me.wav alice.wav --pitch 12
```

```python
from rvc_mlx import RVC, Voice, blend_voices

rvc = RVC.from_pretrained("models/")
alice = Voice.load("models/voices/alice.safetensors")
audio, sr = rvc.convert_file("me.wav", "alice.wav", alice, pitch=12)
```

**→ [Tutorial: converting your voice models](docs/converting-models.md)** (step by step, with troubleshooting).

## Examples

| | |
| --- | --- |
| [**Local web UI**](examples/web-server) | Record or drop audio in the browser, pick a voice, convert. `python examples/web-server/server.py --models models/` |
| [**RVC Pocket** (iOS)](examples/ios) | SwiftUI app: record → pick a voice → convert, fully on-device. Built on **RVCKit**, a Swift package for mlx-swift. |
| [**RVC Morph** (JUCE plug-in)](examples/juce-plugin) | AU / VST3 / Standalone with **ARA 2** offline rendering, a live mode, and an A↔B voice morph slider. Built on a C++ MLX engine. |

## How the pieces fit

```
rvc_mlx/            Python package (reference implementation + converters + CLI)
  hubert.py         HuBERT / ContentVec
  rmvpe.py          RMVPE pitch estimator
  synthesizer.py    RVC v1/v2 synthesizers (relative-attention prior, flow, NSF-HiFiGAN)
  index.py          exact kNN retrieval on MLX (replaces faiss at runtime)
  pipeline.py       the conversion pipeline        voice.py  voice files + blending
  convert.py        .pt / .pth / .index -> .safetensors          cli.py  `rvc-mlx`
examples/
  juce-plugin/engine/   C++ port (MLX C++ API), parity-tested against the Python package
  ios/RVCKit/           Swift port (mlx-swift) of the C++ engine, same parity tests
```

All three runtimes read the same files (format described in the [tutorial appendix](docs/converting-models.md#appendix-the-file-format)).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # Python suite (PyTorch references, end-to-end pipeline, CLI, web example)
```

Parity tests for the other runtimes (also run in CI on Apple silicon):

```bash
python examples/juce-plugin/engine/tests/make_golden.py /tmp/rvc-golden
cmake -B build-engine -DRVC_ENGINE_TESTS=ON -DCMAKE_PREFIX_PATH=$(python -m mlx --cmake-dir) examples/juce-plugin/engine
cmake --build build-engine && ./build-engine/rvc_parity_test /tmp/rvc-golden
```

Linux note: MLX's CPU backend works for running the tests (`tests/conftest.py` disables MLX graph compilation, whose
JIT fails with some GCC versions), but it is far slower than Metal. Use an Apple-silicon Mac for real conversions.

### Test framework

Module tests compare MLX implementations against PyTorch references:

```python
from tests.mlx_torch_comparison_framework import BaseOperationTest, OperationTestSuite

class TestMyOperation(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        cls.suite = OperationTestSuite(my_mlx_func, torch_func, "my_op")
        cls.suite.add_test_case(name="basic_test", inputs={"x": np.array([1, 2, 3])})
```

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgments

- RVC from [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- Built with [MLX](https://github.com/ml-explore/mlx) and [mlx-swift](https://github.com/ml-explore/mlx-swift)
- Plug-in built with [JUCE](https://juce.com) and the [ARA SDK](https://github.com/Celemony/ARA_SDK)
