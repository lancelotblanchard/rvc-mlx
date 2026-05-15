import pytest
import numpy as np
from rvc_mlx.windows import general_cosine, general_hamming, hamming, hann
import torch

from .mlx_torch_comparison_framework import (
    OperationTestSuite,
    BaseOperationTest,
)


class TestWindowsGeneralCosine(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        cls.suite = OperationTestSuite(
            general_cosine, torch.signal.windows.general_cosine, "general_cosine"
        )

        cls.suite.add_test_case(
            name="general_cosine_m0",
            inputs={
                "M": 0,
                "a": [1.0],
            },
            description="General cosine window with M=0",
        )

        cls.suite.add_test_case(
            name="general_cosine_m1",
            inputs={
                "M": 1,
                "a": [1.0],
            },
            description="General cosine window with M=1",
        )

        cls.suite.add_test_case(
            name="general_cosine_3_coefficients",
            inputs={
                "M": 10,
                "a": [0.46, 0.23, 0.31],
                "sym": True,
            },
            description="General cosine window with 3 coefficients",
        )

        cls.suite.add_test_case(
            name="general_cosine_2_coefficients_sym_false",
            inputs={
                "M": 10,
                "a": [0.5, 1 - 0.5],
                "sym": False,
            },
            description="General cosine periodic window with 2 coefficients",
        )


class TestWindowsGeneralHamming(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        cls.suite = OperationTestSuite(
            general_hamming, torch.signal.windows.general_hamming, "general_hamming"
        )

        cls.suite.add_test_case(
            name="general_hamming_default",
            inputs={"M": 16},
            description="General Hamming window with default alpha and sym=True",
        )

        cls.suite.add_test_case(
            name="general_hamming_custom_alpha",
            inputs={"M": 32, "alpha": 0.4},
            description="General Hamming window with custom alpha",
        )

        cls.suite.add_test_case(
            name="general_hamming_periodic",
            inputs={"M": 64, "alpha": 0.54, "sym": False},
            description="Periodic general Hamming window (sym=False)",
        )


class TestWindowsHamming(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        cls.suite = OperationTestSuite(
            hamming, torch.signal.windows.hamming, "hamming"
        )

        cls.suite.add_test_case(
            name="hamming_default",
            inputs={"M": 16},
            description="Hamming window with default sym=True",
        )

        cls.suite.add_test_case(
            name="hamming_periodic",
            inputs={"M": 32, "sym": False},
            description="Periodic Hamming window (sym=False)",
        )

        cls.suite.add_test_case(
            name="hamming_large",
            inputs={"M": 1024, "sym": True},
            description="Symmetric Hamming window of typical STFT size",
        )


class TestWindowsHann(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        cls.suite = OperationTestSuite(
            hann, torch.signal.windows.hann, "hann"
        )

        cls.suite.add_test_case(
            name="hann_default",
            inputs={"M": 16},
            description="Hann window with default sym=True",
        )

        cls.suite.add_test_case(
            name="hann_periodic",
            inputs={"M": 32, "sym": False},
            description="Periodic Hann window (sym=False, matches torch.hann_window default)",
        )

        cls.suite.add_test_case(
            name="hann_large",
            inputs={"M": 1024, "sym": True},
            description="Symmetric Hann window of typical STFT size",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
