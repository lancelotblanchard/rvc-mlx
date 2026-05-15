"""
Example test file showing how to use the MLX testing framework with pytest.

Run with:
    pytest test_mlx_operations.py -v
    pytest test_mlx_operations.py::TestReLU -v  # Run only ReLU tests
    pytest test_mlx_operations.py -v -k "negative"  # Run tests matching pattern
"""
from functools import partial

import pytest
import numpy as np
from rvc_mlx.utils import narrow, pad_constant, sequence_mask
import torch

from .mlx_torch_comparison_framework import (
    OperationTestSuite, BaseOperationTest,
)

class TestUtilsNarrow(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        cls.suite = OperationTestSuite(narrow, torch.narrow, "narrow")

        test_array = np.arange(24).reshape(2, 3, 4).astype(np.float32)

        cls.suite.add_test_case(
            name="narrow_dim0",
            inputs={'x': test_array, 'dim': 0, 'start': 0, 'length': 1},
            description="Narrow along first dimension"
        )

        cls.suite.add_test_case(
            name="narrow_dim1",
            inputs={'x': test_array, 'dim': 1, 'start': 1, 'length': 2},
            description="Narrow along second dimension"
        )

        cls.suite.add_test_case(
            name="narrow_dim2",
            inputs={'x': test_array, 'dim': 2, 'start': 0, 'length': 3},
            description="Narrow along third dimension"
        )

        cls.suite.add_test_case(
            name="narrow_negative_start",
            inputs={'x': test_array, 'dim': 0, 'start': -1, 'length': 1},
            description="Narrow with negative start"
        )

        cls.suite.add_test_case(
            name="narrow_negative_dimension",
            inputs={'x': test_array, 'dim': -1, 'start': 0, 'length': 2},
            description="Narrow with negative dimension"
        )

        cls.suite.add_test_case(
            name="narrow_dim_out_of_range",
            inputs={'x': test_array, 'dim': 3, 'start': 0, 'length': 3},
            should_error=True,
            error_type=IndexError,
            description="Should error when dimension is out of range"
        )

        cls.suite.add_test_case(
            name="narrow_negative_length",
            inputs={'x': test_array, 'dim': 0, 'start': 0, 'length': -1},
            should_error=True,
            error_type=ValueError,
            description="Should error when length is negative"
        )

        cls.suite.add_test_case(
            name="narrow_start_out_of_bounds",
            inputs={'x': test_array, 'dim': 0, 'start': 5, 'length': 1},
            should_error=True,
            error_type=ValueError,
            description="Should error when start index is out of bounds"
        )

        cls.suite.add_test_case(
            name="narrow_length_out_of_bounds",
            inputs={'x': test_array, 'dim': 0, 'start': 0, 'length': 5},
            should_error=True,
            error_type=ValueError,
            description="Should error when length is out of bounds"
        )

class TestUtilsPadConstant(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        torch_pad_constant = partial(torch.nn.functional.pad, mode='constant')

        cls.suite = OperationTestSuite(pad_constant, torch_pad_constant, "pad_constant")

        # Basic positive padding tests
        cls.suite.add_test_case(
            name="1d_symmetric_pad",
            inputs={
                'input': np.array([1.0, 2.0, 3.0]),
                'pad': (1, 1),
                'value': 0.0
            },
            description="Pad 1D array symmetrically with zeros"
        )

        cls.suite.add_test_case(
            name="1d_asymmetric_pad",
            inputs={
                'input': np.array([1.0, 2.0, 3.0]),
                'pad': (2, 1),
                'value': 0.0
            },
            description="Pad 1D array asymmetrically"
        )

        cls.suite.add_test_case(
            name="1d_pad_with_value",
            inputs={
                'input': np.array([1.0, 2.0, 3.0]),
                'pad': (1, 1),
                'value': -1.0
            },
            description="Pad 1D array with custom value"
        )

        # 2D padding tests
        test_2d = np.arange(12).reshape(3, 4).astype(np.float32)

        cls.suite.add_test_case(
            name="2d_pad_last_dim",
            inputs={
                'input': test_2d,
                'pad': (1, 1),
                'value': 0.0
            },
            description="Pad only last dimension of 2D array"
        )

        cls.suite.add_test_case(
            name="2d_pad_both_dims",
            inputs={
                'input': test_2d,
                'pad': (1, 1, 2, 2),
                'value': 0.0
            },
            description="Pad both dimensions of 2D array"
        )

        cls.suite.add_test_case(
            name="2d_asymmetric_pad",
            inputs={
                'input': test_2d,
                'pad': (1, 2, 0, 3),
                'value': 0.0
            },
            description="Asymmetric padding on 2D array"
        )

        # 3D padding tests
        test_3d = np.arange(24).reshape(2, 3, 4).astype(np.float32)

        cls.suite.add_test_case(
            name="3d_pad_last_dim",
            inputs={
                'input': test_3d,
                'pad': (1, 1),
                'value': 0.0
            },
            description="Pad only last dimension of 3D array"
        )

        cls.suite.add_test_case(
            name="3d_pad_two_dims",
            inputs={
                'input': test_3d,
                'pad': (1, 1, 2, 2),
                'value': 0.0
            },
            description="Pad last two dimensions of 3D array"
        )

        cls.suite.add_test_case(
            name="3d_pad_all_dims",
            inputs={
                'input': test_3d,
                'pad': (1, 1, 1, 1, 1, 1),
                'value': 0.0
            },
            description="Pad all dimensions of 3D array"
        )

        # Negative padding (narrowing) tests
        cls.suite.add_test_case(
            name="1d_negative_pad_left",
            inputs={
                'input': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                'pad': (-1, 0),
                'value': 0.0
            },
            description="Narrow 1D array from left (negative padding)"
        )

        cls.suite.add_test_case(
            name="1d_negative_pad_right",
            inputs={
                'input': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                'pad': (0, -1),
                'value': 0.0
            },
            description="Narrow 1D array from right (negative padding)"
        )

        cls.suite.add_test_case(
            name="1d_negative_pad_both",
            inputs={
                'input': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                'pad': (-1, -1),
                'value': 0.0
            },
            description="Narrow 1D array from both sides"
        )

        cls.suite.add_test_case(
            name="2d_negative_pad",
            inputs={
                'input': test_2d,
                'pad': (-1, -1, -1, -1),
                'value': 0.0
            },
            description="Narrow 2D array on all sides"
        )

        # Mixed positive and negative padding
        cls.suite.add_test_case(
            name="1d_mixed_pad",
            inputs={
                'input': np.array([1.0, 2.0, 3.0, 4.0]),
                'pad': (-1, 2),
                'value': 0.0
            },
            description="Mix of narrowing and padding"
        )

        cls.suite.add_test_case(
            name="2d_mixed_pad",
            inputs={
                'input': test_2d,
                'pad': (1, -1, -1, 2),
                'value': 0.0
            },
            description="2D array with mixed positive/negative padding"
        )

        # Zero padding (no-op)
        cls.suite.add_test_case(
            name="1d_zero_pad",
            inputs={
                'input': np.array([1.0, 2.0, 3.0]),
                'pad': (0, 0),
                'value': 0.0
            },
            description="Zero padding (should return unchanged)"
        )

        cls.suite.add_test_case(
            name="2d_zero_pad",
            inputs={
                'input': test_2d,
                'pad': (0, 0, 0, 0),
                'value': 0.0
            },
            description="Zero padding on 2D array"
        )

        # Edge cases with different values
        cls.suite.add_test_case(
            name="pad_with_large_value",
            inputs={
                'input': np.ones((3, 3)),
                'pad': (1, 1, 1, 1),
                'value': 999.0
            },
            description="Padding with large constant value"
        )

        cls.suite.add_test_case(
            name="pad_with_negative_value",
            inputs={
                'input': np.ones((3, 3)),
                'pad': (1, 1, 1, 1),
                'value': -100.0
            },
            description="Padding with negative constant value"
        )

        cls.suite.add_test_case(
            name="pad_with_different_type_value",
            inputs={
                'input': np.array([1.0, 2.0, 3.0]),
                'pad': (2, 2),
                'value': 0,
            }
        )

        # Error cases
        cls.suite.add_test_case(
            name="odd_length_pad",
            inputs={
                'input': np.array([1.0, 2.0, 3.0]),
                'pad': (1, 1, 1),
                'value': 0.0
            },
            should_error=True,
            error_type=ValueError,
            description="Should error when pad length is odd"
        )

        cls.suite.add_test_case(
            name="too_many_pad_dims",
            inputs={
                'input': np.array([1.0, 2.0, 3.0]),
                'pad': (1, 1, 1, 1, 1, 1),
                'value': 0.0
            },
            should_error=True,
            error_type=ValueError,
            description="Should error when pad dimensions exceed input dimensions"
        )

        cls.suite.add_test_case(
            name="negative_output_size",
            inputs={
                'input': np.array([1.0, 2.0, 3.0]),
                'pad': (-5, -5),
                'value': 0.0
            },
            should_error=True,
            error_type=ValueError,
            description="Should error when negative padding results in negative output size"
        )

        # Large padding values
        cls.suite.add_test_case(
            name="large_positive_pad",
            inputs={
                'input': np.array([1.0, 2.0]),
                'pad': (10, 10),
                'value': 0.0
            },
            description="Large positive padding values"
        )

        # Single element array
        cls.suite.add_test_case(
            name="single_element_pad",
            inputs={
                'input': np.array([5.0]),
                'pad': (2, 3),
                'value': 0.0
            },
            description="Pad single element array"
        )

        # Different data types
        cls.suite.add_test_case(
            name="integer_input",
            inputs={
                'input': np.array([1, 2, 3, 4], dtype=np.int32),
                'pad': (1, 1),
                'value': 0
            },
            description="Padding with integer input"
        )

def _torch_sequence_mask(length, max_length=None):
    """Reference implementation matching the helper used in RVC."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class TestUtilsSequenceMask(BaseOperationTest):
    @classmethod
    def setup_class(cls):
        cls.suite = OperationTestSuite(sequence_mask, _torch_sequence_mask, "sequence_mask")

        cls.suite.add_test_case(
            name="sequence_mask_implicit_max",
            inputs={"length": np.array([1, 3, 2], dtype=np.int32)},
            description="Sequence mask with max_length inferred from input",
        )

        cls.suite.add_test_case(
            name="sequence_mask_explicit_max",
            inputs={"length": np.array([2, 4, 1], dtype=np.int32), "max_length": 5},
            description="Sequence mask with explicit max_length greater than max(length)",
        )

        cls.suite.add_test_case(
            name="sequence_mask_max_equals_length",
            inputs={"length": np.array([3, 3, 3], dtype=np.int32), "max_length": 3},
            description="Sequence mask with max_length equal to all lengths (all True)",
        )

        cls.suite.add_test_case(
            name="sequence_mask_zero_length",
            inputs={"length": np.array([0, 2, 0], dtype=np.int32), "max_length": 3},
            description="Sequence mask with zero-length entries (rows of all False)",
        )

        cls.suite.add_test_case(
            name="sequence_mask_single_batch",
            inputs={"length": np.array([5], dtype=np.int32), "max_length": 7},
            description="Sequence mask with a single element in the batch",
        )

        cls.suite.add_test_case(
            name="sequence_mask_large",
            inputs={"length": np.array([100, 50, 75, 128], dtype=np.int64), "max_length": 128},
            description="Sequence mask with larger lengths and int64 dtype",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])