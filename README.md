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

```python
from rvc_mlx import your_function

# Your usage example
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
