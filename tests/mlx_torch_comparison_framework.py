"""
Claude-generated

MLX vs PyTorch Testing Framework

A comprehensive framework for comparing MLX implementations against PyTorch reference implementations.

Installation:
    pip install pytest pytest-xdist numpy mlx torch

Usage:
    pytest test_mlx_operations.py -v
    pytest test_mlx_operations.py -v -k "test_relu"  # Run specific test
    pytest test_mlx_operations.py -v -n auto  # Parallel execution
"""

import pytest
import numpy as np
from typing import Callable, Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TestType(Enum):
    """Types of tests to run"""
    OUTPUT = "output"  # Compare outputs
    ERROR = "error"    # Compare error handling
    SHAPE = "shape"    # Compare output shapes
    DTYPE = "dtype"    # Compare data types


@dataclass
class TestCase:
    """Defines a single test case with inputs and expected behavior"""
    name: str
    inputs: Dict[str, Any]
    test_types: List[TestType] = None
    rtol: float = 1e-5
    atol: float = 1e-5
    should_error: bool = False
    error_type: Optional[type] = None
    description: str = ""

    def __post_init__(self):
        if self.test_types is None:
            self.test_types = [TestType.OUTPUT, TestType.SHAPE, TestType.DTYPE]


class OperationTestSuite:
    """Base class for testing MLX operations against PyTorch references"""

    def __init__(
        self,
        mlx_func: Callable,
        torch_func: Callable,
        name: str = None
    ):
        """
        Args:
            mlx_func: Your MLX implementation
            torch_func: PyTorch reference implementation
            name: Optional name for the operation (defaults to function name)
        """
        self.mlx_func = mlx_func
        self.torch_func = torch_func
        self.name = name or mlx_func.__name__
        self.test_cases: List[TestCase] = []

    def add_test_case(
        self,
        name: str,
        inputs: Dict[str, Any],
        test_types: List[TestType] = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        should_error: bool = False,
        error_type: Optional[type] = None,
        description: str = ""
    ):
        """Add a test case to the suite"""
        test_case = TestCase(
            name=name,
            inputs=inputs,
            test_types=test_types,
            rtol=rtol,
            atol=atol,
            should_error=should_error,
            error_type=error_type,
            description=description
        )
        self.test_cases.append(test_case)
        return self

    def _convert_to_mlx(self, obj):
        """Convert numpy arrays or tensors to MLX arrays"""
        import mlx.core as mx
        if isinstance(obj, np.ndarray):
            return mx.array(obj)
        elif hasattr(obj, 'numpy'):  # torch tensor
            return mx.array(obj.numpy())
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._convert_to_mlx(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_mlx(v) for k, v in obj.items()}
        return obj

    def _convert_to_torch(self, obj):
        """Convert numpy arrays to PyTorch tensors"""
        import torch
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._convert_to_torch(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_torch(v) for k, v in obj.items()}
        return obj

    def _to_numpy(self, obj):
        """Convert MLX array or PyTorch tensor to numpy"""
        if hasattr(obj, 'numpy'):  # torch tensor or similar
            return obj.numpy()
        elif hasattr(obj, '__array__'):  # numpy-like (including mx.array)
            return np.array(obj)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._to_numpy(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._to_numpy(v) for k, v in obj.items()}
        return obj

    def _compare_arrays(self, mlx_result, torch_result, rtol, atol) -> Tuple[bool, str]:
        """Compare two arrays with tolerance"""
        mlx_np = self._to_numpy(mlx_result)
        torch_np = self._to_numpy(torch_result)

        # Both must be numpy arrays at this point, if not treat as arrays
        if not isinstance(mlx_np, np.ndarray):
            mlx_np = np.asarray(mlx_np)
        if not isinstance(torch_np, np.ndarray):
            torch_np = np.asarray(torch_np)

        if mlx_np.shape != torch_np.shape:
            return False, f"Shape mismatch: MLX {mlx_np.shape} vs PyTorch {torch_np.shape}"

        if mlx_np.dtype != torch_np.dtype:
            # Allow some dtype flexibility (e.g., float32 vs float64)
            if not (np.issubdtype(mlx_np.dtype, np.floating) and
                    np.issubdtype(torch_np.dtype, np.floating)):
                return False, f"Dtype mismatch: MLX {mlx_np.dtype} vs PyTorch {torch_np.dtype}"

        try:
            np.testing.assert_allclose(mlx_np, torch_np, rtol=rtol, atol=atol)
            return True, ""
        except AssertionError as e:
            max_diff = np.max(np.abs(mlx_np - torch_np))
            return False, f"Values differ (max diff: {max_diff}): {str(e)}"

    def run_test_case(self, test_case: TestCase) -> Dict[str, Any]:
        """Run a single test case and return results"""
        results = {
            'passed': True,
            'errors': [],
            'test_name': test_case.name,
            'description': test_case.description
        }

        # Prepare inputs
        mlx_inputs = self._convert_to_mlx(test_case.inputs)
        torch_inputs = self._convert_to_torch(test_case.inputs)

        # Test error handling if expected
        if test_case.should_error:
            mlx_error = None
            torch_error = None

            try:
                self.mlx_func(**mlx_inputs)
            except Exception as e:
                mlx_error = e

            try:
                self.torch_func(**torch_inputs)
            except Exception as e:
                torch_error = e

            if mlx_error is None:
                results['passed'] = False
                results['errors'].append("MLX function did not raise expected error")

            if torch_error is None:
                results['passed'] = False
                results['errors'].append("PyTorch function did not raise expected error")

            if test_case.error_type and not isinstance(mlx_error, test_case.error_type):
                results['passed'] = False
                results['errors'].append(
                    f"MLX error type mismatch: expected {test_case.error_type}, "
                    f"got {type(mlx_error)}"
                )

            return results

        # Run both implementations
        try:
            mlx_result = self.mlx_func(**mlx_inputs)
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"MLX function raised error: {e}")
            return results

        try:
            torch_result = self.torch_func(**torch_inputs)
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"PyTorch function raised error: {e}")
            return results

        # Compare results based on test types
        if TestType.OUTPUT in test_case.test_types:
            passed, msg = self._compare_arrays(
                mlx_result, torch_result, test_case.rtol, test_case.atol
            )
            if not passed:
                results['passed'] = False
                results['errors'].append(f"Output comparison failed: {msg}")

        if TestType.SHAPE in test_case.test_types:
            mlx_shape = getattr(mlx_result, 'shape', None)
            torch_shape = getattr(torch_result, 'shape', None)
            if mlx_shape != torch_shape:
                results['passed'] = False
                results['errors'].append(
                    f"Shape mismatch: MLX {mlx_shape} vs PyTorch {torch_shape}"
                )

        if TestType.DTYPE in test_case.test_types:
            mlx_dtype = getattr(mlx_result, 'dtype', None)
            torch_dtype = getattr(torch_result, 'dtype', None)
            if mlx_dtype != torch_dtype:
                # Allow some flexibility in dtype comparison
                results['errors'].append(
                    f"Dtype difference: MLX {mlx_dtype} vs PyTorch {torch_dtype}"
                )

        return results

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all test cases"""
        return [self.run_test_case(tc) for tc in self.test_cases]


class BaseOperationTest:
    """
    Abstract base class for operation tests.

    Subclasses should:
    1. Define `suite` as a class attribute (typically in setup_class)
    2. Optionally override test_operation if custom behavior is needed

    Example:
        class TestMyOperation(BaseOperationTest):
            @classmethod
            def setup_class(cls):
                cls.suite = OperationTestSuite(mlx_func, torch_func, "my_op")
                cls.suite.add_test_case(...)
    """

    suite: OperationTestSuite = None

    def test_operation(self):
        """Run all test cases in the suite"""
        if self.suite is None:
            pytest.skip("No test suite defined")

        for test_case in self.suite.test_cases:
            results = self.suite.run_test_case(test_case)
            if not results['passed']:
                error_msg = f"\nTest: {results['test_name']}\n"
                if results['description']:
                    error_msg += f"Description: {results['description']}\n"
                error_msg += "Errors:\n" + "\n".join(f"  - {e}" for e in results['errors'])
                pytest.fail(error_msg)


# Pytest integration
def create_pytest_tests(test_suite: OperationTestSuite):
    """Generate pytest test functions from a test suite"""

    @pytest.mark.parametrize("test_case", test_suite.test_cases, ids=lambda tc: tc.name)
    def test_operation(test_case):
        """Parametrized test function"""
        results = test_suite.run_test_case(test_case)

        if not results['passed']:
            error_msg = f"\nTest: {results['test_name']}\n"
            if results['description']:
                error_msg += f"Description: {results['description']}\n"
            error_msg += "Errors:\n" + "\n".join(f"  - {e}" for e in results['errors'])
            pytest.fail(error_msg)

    return test_operation
