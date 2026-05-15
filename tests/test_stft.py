from functools import partial

import pytest
import numpy as np
from rvc_mlx.stft import stft
import torch

from .mlx_torch_comparison_framework import (
    OperationTestSuite,
    BaseOperationTest,
)


class TestUtilsStft(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        torch_stft_constant = partial(
            torch.stft, return_complex=True, pad_mode="constant", onesided=False
        )

        cls.suite = OperationTestSuite(stft, torch_stft_constant, "stft")

        cls.suite.add_test_case(
            name="stft_basic",
            inputs={
                "input": np.random.random_sample((1024,)),
                "n_fft": 512,
            },
            description="Compute STFT with n_fft parameter",
            atol=1e-4,
            rtol=1e-4,
        )

        cls.suite.add_test_case(
            name="stft_batch",
            inputs={
                "input": np.random.random_sample((5, 1024)),
                "n_fft": 512,
            },
            description="Compute STFT with n_fft parameter and additional batch channel",
            atol=1e-4,
            rtol=1e-4,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
