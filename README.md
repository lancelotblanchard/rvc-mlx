# RVC-MLX

[![Tests](https://github.com/lancelotblanchard/rvc-mlx/actions/workflows/tests.yml/badge.svg)](https://github.com/lancelotblanchard/rvc-mlx/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/github/lancelotblanchard/rvc-mlx/graph/badge.svg?token=78POKEWDJ3)](https://codecov.io/github/lancelotblanchard/rvc-mlx)

> Your project description here

## Features

- 🚀 Fast MLX implementations of [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- ✅ Comprehensive test suite comparing to PyTorch

## Installation

```bash
pip install rvc-mlx
```

Or install from source:

```bash
git clone https://github.com/lancelotblanchard/rvc-mlx.git
cd rvc-mlx
pip install -e .
```

## Usage

Run a voice conversion end-to-end from the command line:

```bash
rvc-mlx \
    --input source.wav \
    --output converted.wav \
    --voice path/to/voice.pth \
    --hubert path/to/hubert_base.pt \
    --rmvpe path/to/rmvpe.pt \
    --speaker-id 0 \
    --pitch-shift 0 \
    --rms-mix-rate 0.25
```

Or from Python:

```python
from rvc_mlx.infer import run_inference

run_inference(
    input_path="source.wav",
    output_path="converted.wav",
    voice="path/to/voice.pth",
    hubert="path/to/hubert_base.pt",
    rmvpe="path/to/rmvpe.pt",
    speaker_id=0,
    pitch_shift=0,
)
```

Released `.pth` / `.pt` checkpoints (RVC voices, fairseq HuBERT, RMVPE) and our native `.safetensors`
(with sibling `.config.json`) are both accepted. The first call against a `.pth` / `.pt` writes a converted
`.safetensors` next to the original; subsequent loads skip the conversion (and skip importing PyTorch).

Lower-level Python API for finer control:

```python
from types import SimpleNamespace

from rvc_mlx.audio import load_audio_16k, save_audio
from rvc_mlx.hubert import HubertModel
from rvc_mlx.pipeline import Pipeline
from rvc_mlx.rmvpe import RMVPE
from rvc_mlx.synthesizer import SynthesizerTrnMs768NSFsid

hubert = HubertModel.from_pretrained("hubert_base.pt")
rmvpe = RMVPE.from_pretrained("rmvpe.pt")
voice = SynthesizerTrnMs768NSFsid.from_pretrained("voice.pth")

cfg = SimpleNamespace(x_pad=1, x_query=6, x_center=38, x_max=41, is_half=False,
                     rmvpe_root=None, hubert_root=None)
pipe = Pipeline(tgt_sr=voice.sr, config=cfg, rmvpe=rmvpe, hubert=hubert)

audio = load_audio_16k("source.wav")
converted = pipe.pipeline(voice, audio, sid=0, f0_up_key=0)
save_audio("converted.wav", converted, voice.sr)
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/lancelotblanchard/rvc-mlx.git
cd rvc-mlx

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rvc_mlx --cov-report=html

# Run specific test file
pytest tests/test_utils.py -v

# Run tests matching pattern
pytest -k "narrow" -v

# Run in parallel
pytest -n auto
```

### Test Framework

This project uses a custom testing framework that compares MLX implementations against PyTorch references:

```python
from tests.mlx_torch_comparison_framework import BaseOperationTest, OperationTestSuite

class TestMyOperation(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        cls.suite = OperationTestSuite(my_mlx_func, torch_func, "my_op")
        cls.suite.add_test_case(
            name="basic_test",
            inputs={'x': np.array([1, 2, 3])},
            description="Test basic functionality"
        )
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RVC Implementation from [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- Built with [MLX](https://github.com/ml-explore/mlx)
- Reference implementations from [PyTorch](https://pytorch.org/)
