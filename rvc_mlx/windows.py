import mlx.core as mx
from typing import Iterable, Optional


def _window_function_checks(
    function_name: str,
    M: int,
    dtype: mx.Dtype,
) -> None:
    if M < 0:
        raise ValueError(f"{function_name} requires non-negative window length, got M={M}")
    if dtype not in [mx.float32, mx.float64]:
        raise ValueError(f"{function_name} expects float32 or float64 dtypes, got: {dtype}")


def general_cosine(
    M: int,
    a: Iterable,
    sym: bool = False,
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    if dtype is None:
        dtype = mx.float32

    _window_function_checks("general_cosine", M, dtype)

    if M == 0:
        return mx.zeros((0,), dtype=dtype)

    if M == 1:
        return mx.ones((1,), dtype=dtype)

    if not isinstance(a, Iterable):
        raise TypeError("Coefficients must be a list/tuple")

    if not a:
        raise ValueError("Coefficients cannot be empty")

    constant = 2 * mx.pi / (M if not sym else M - 1)

    k = mx.linspace(
        start=0,
        stop=(M - 1) * constant,
        num=M,
        dtype=dtype,
    )

    a_i = mx.array(
        [(-1) ** i * w for i, w in enumerate(a)],
        dtype=dtype,
    )
    i = mx.arange(
        a_i.shape[0],
        dtype=a_i.dtype,
    )
    return (mx.expand_dims(a_i, -1) * mx.cos(mx.expand_dims(i, -1) * k)).sum(0)


def general_hamming(
    M: int,
    alpha: float = 0.54,
    sym: bool = True,
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    return general_cosine(
        M,
        a=[alpha, 1.0 - alpha],
        sym=sym,
        dtype=dtype,
    )


def hamming(
    M: int,
    sym: bool = True,
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    return general_hamming(
        M,
        sym=sym,
        dtype=dtype,
    )


def hann(
    M: int,
    sym: bool = True,
    dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    return general_hamming(
        M,
        alpha=0.5,
        sym=sym,
        dtype=dtype,
    )
