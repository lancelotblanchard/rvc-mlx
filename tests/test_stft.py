from functools import partial

import pytest
import numpy as np
from rvc_mlx.stft import stft
import torch

from .mlx_torch_comparison_framework import (
    OperationTestSuite,
    BaseOperationTest,
)


# Cast inputs to float32 so the framework does not promote to float64 inside MLX (which would surface as a complex64 vs
# complex128 dtype mismatch in the comparison). Real audio in RVC is always float32.
_RNG = np.random.default_rng(0)


class TestUtilsStftConstantTwoSided(BaseOperationTest):
    """STFT with the legacy `pad_mode="constant"`, `onesided=False` configuration."""

    @classmethod
    def setup_class(cls):
        mlx_stft = partial(stft, pad_mode="constant", onesided=False)
        torch_stft = partial(
            torch.stft, return_complex=True, pad_mode="constant", onesided=False
        )
        cls.suite = OperationTestSuite(mlx_stft, torch_stft, "stft_constant_twosided")

        cls.suite.add_test_case(
            name="stft_basic",
            inputs={
                "input": _RNG.standard_normal((1024,)).astype(np.float32),
                "n_fft": 512,
            },
            description="Compute STFT with n_fft parameter",
            atol=1e-4,
            rtol=1e-4,
        )

        cls.suite.add_test_case(
            name="stft_batch",
            inputs={
                "input": _RNG.standard_normal((5, 1024)).astype(np.float32),
                "n_fft": 512,
            },
            description="Compute STFT with n_fft parameter and additional batch channel",
            atol=1e-4,
            rtol=1e-4,
        )


class TestUtilsStftReflectTwoSided(BaseOperationTest):
    """STFT with `pad_mode="reflect"` (MLX and torch default), `onesided=False`."""

    @classmethod
    def setup_class(cls):
        mlx_stft = partial(stft, pad_mode="reflect", onesided=False)
        torch_stft = partial(
            torch.stft, return_complex=True, pad_mode="reflect", onesided=False
        )
        cls.suite = OperationTestSuite(mlx_stft, torch_stft, "stft_reflect_twosided")

        cls.suite.add_test_case(
            name="stft_reflect_1d",
            inputs={
                "input": _RNG.standard_normal((2048,)).astype(np.float32),
                "n_fft": 512,
                "hop_length": 128,
            },
            description="1D STFT with reflect padding (the torch default)",
            atol=1e-4,
            rtol=1e-4,
        )

        cls.suite.add_test_case(
            name="stft_reflect_batched",
            inputs={
                "input": _RNG.standard_normal((3, 2048)).astype(np.float32),
                "n_fft": 512,
                "hop_length": 128,
            },
            description="Batched STFT with reflect padding",
            atol=1e-4,
            rtol=1e-4,
        )


class TestUtilsStftOnesided(BaseOperationTest):
    """STFT with `onesided=True` (the torch default for real input) under both padding modes."""

    @classmethod
    def setup_class(cls):
        mlx_stft = partial(stft, pad_mode="reflect", onesided=True)
        torch_stft = partial(
            torch.stft, return_complex=True, pad_mode="reflect", onesided=True
        )
        cls.suite = OperationTestSuite(mlx_stft, torch_stft, "stft_onesided")

        cls.suite.add_test_case(
            name="onesided_basic",
            inputs={
                "input": _RNG.standard_normal((2048,)).astype(np.float32),
                "n_fft": 512,
            },
            description="One-sided STFT keeps only n_fft//2 + 1 frequency bins",
            atol=1e-4,
            rtol=1e-4,
        )

        cls.suite.add_test_case(
            name="onesided_batched",
            inputs={
                "input": _RNG.standard_normal((2, 4000)).astype(np.float32),
                "n_fft": 1024,
                "hop_length": 160,
            },
            description="Batched one-sided STFT with RVC-shaped framing",
            atol=1e-4,
            rtol=1e-4,
        )


class TestUtilsStftWithWindow(BaseOperationTest):
    """STFT with an explicit Hann window and onesided=True, reflect pad."""

    @classmethod
    def setup_class(cls):
        win = torch.hann_window(400)  # periodic Hann, matches our hann(400, sym=False)
        win_mlx = win.numpy()  # numpy; framework will convert per-side
        mlx_stft = partial(stft, pad_mode="reflect", onesided=True)
        torch_stft = partial(
            torch.stft, return_complex=True, pad_mode="reflect", onesided=True
        )
        cls.suite = OperationTestSuite(mlx_stft, torch_stft, "stft_with_window")

        cls.suite.add_test_case(
            name="window_explicit",
            inputs={
                "input": _RNG.standard_normal((1, 8000)).astype(np.float32),
                "n_fft": 512,
                "hop_length": 128,
                "win_length": 400,
                "window": win_mlx.astype(np.float32),
            },
            description="STFT with an explicit window shorter than n_fft (zero-padded internally)",
            atol=1e-4,
            rtol=1e-4,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
