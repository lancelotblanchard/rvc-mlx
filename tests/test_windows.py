import pytest
import numpy as np
from rvc_mlx.windows import general_cosine
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
