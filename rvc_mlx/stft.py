import mlx.core as mx
from typing import Optional

from rvc_mlx.utils import pad_constant


def stft(
    input: mx.array,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[mx.array] = None,
    center: bool = True,
    pad_mode: str = "constant",
    normalized: bool = False,
    onesided: bool = False,
    return_complex: bool = True,
):
    """
    The STFT computes the Fourier transform of short overlapping windows of the input. This giving frequency components
    of the signal as they change over time. In agreement with future releases of PyTorch, the STFT is always returned
    as a complex matrix. We also only support complex inputs, and force normalized=False and onesided=False.

    :param input: the input tensor of shape (B?, L) where B? is an optional batch dimension
    :param n_fft: size of Fourier transform
    :param hop_length: the distance between neighboring sliding window frames. Default: None (treated as equal to
    floor(n_fft / 4))
    :param win_length: the size of window frame and STFT filter. Default: None (treated as equal to n_fft)
    :param window: the optional window function. Shape must be 1d and <= n_fft Default: None (treated as window of all
    1s)
    :param center: whether to pad input on both sides so that the tt-th frame is centered at time t×hop_length. Default:
    True
    :param pad_mode: only "constant" is supported for now TODO: Implement others
    :param normalized: only False is supported
    :param onesided: only False is supported
    :param return_complex: only True is supported
    :return:  A tensor containing the STFT result with shape (B?, N, T) where:
        - B? is an optional batch dimension from the input.
        - N is the number of frequency samples, n_fft.
        - T is the number of frames, 1 + L // hop_length for center=True, or 1 + (L - n_fft) // hop_length otherwise.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if pad_mode != "constant":
        raise NotImplementedError(f"pad_mode {pad_mode} not implemented.")
    if normalized:
        raise ValueError("Cannot pass `normalized=True` to stft.")
    if onesided:
        raise ValueError("Cannot pass `onesided=True` to stft.")
    if not return_complex:
        raise ValueError("Cannot pass `return_complex=False` to stft.")

    x = input

    if not mx.issubdtype(x.dtype, mx.floating) and not mx.issubdtype(x.dtype, mx.complexfloating):
        raise ValueError(f"Expected an array of floating point or complex values, got {x.dtype}")
    if x.ndim not in [1, 2]:
        raise ValueError(f"Expected an array of 1 or 2 dimensions, got {x.ndim}")
    if x.ndim == 1:
        x = mx.expand_dims(x, 0)

    if center:
        x_dim = x.ndim
        extra_dims = max(3, x_dim) - x_dim
        extended_shape = [1] * extra_dims + list(x.shape)
        pad_amount = n_fft // 2
        x = pad_constant(x.reshape(extended_shape), (pad_amount, pad_amount), 0)
        x = x.reshape(x.shape[extra_dims:])

    batch = x.shape[0]
    length = x.shape[1]
    if n_fft <= 0 or n_fft > length:
        return ValueError(f"Expected 0 < n_fft < {len}, but got n_fft={win_length}")
    if hop_length <= 0:
        return ValueError(f"Expected hop_length > 0, but got hop_length={hop_length}")
    if win_length <= 0 or win_length > n_fft:
        return ValueError(f"expected 0 < win_length <= n_fft, but got win_length={win_length}")
    if window is not None and (window.ndim != 1 or window.shape[0] != win_length):
        return ValueError(
            f"Expected a 1D window tensor of size equal to win_length={win_length}, but got window with"
            f"size {window.shape}"
        )

    window_ = window
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        if window is not None:
            window_ = mx.zeros((n_fft,))
            window_ = mx.slice_update(window_, window, start_indices=mx.array([left]), axes=(0,))
        else:
            window_ = mx.ones((n_fft,))
            window_ = pad_constant(window_, (left, 0), value=0)

    n_frames = 1 + (length - n_fft) // hop_length
    x = mx.as_strided(x, (batch, n_frames, n_fft), (length, hop_length, 1))
    if window_ is not None:
        x = x * window_

    out: mx.array = mx.fft.fft(x)  # type: ignore[attr-defined]
    out = out.transpose(0, 2, 1)
    if input.ndim == 1:
        out = out.squeeze(0)

    return out
