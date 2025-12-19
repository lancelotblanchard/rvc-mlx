# import mlx.core as mx
# from typing import Optional

# TODO: Continue based on https://github.com/pytorch/pytorch/blob/eba9265a580c6dc3e928ef341c23cab96ccf8b07/aten/src/ATen/native/SpectralOps.cpp#L825
# def stft(
#     x: mx.array,
#     n_fft: int,
#     hop_length: Optional[int] = None,
#     win_length: Optional[int] = None,
#     center = True,
# ):
#     """
#     In agreement with future releases of PyTorch, the STFT
#     is always returned as a complex matrix.
#     :param n_fft:
#     :param hop_length:
#     :param win_length:
#     :return:
#     """
#     if hop_length is None:
#         hop_length = n_fft // 2
#     if win_length is None:
#         win_length = n_fft
#
#     # TODO: Change to ValueError
#     assert mx.issubdtype(x.dtype, mx.floating) or mx.issubdtype(x.dtype, mx.complexfloating), (
#         "Expected an array of floating point or complex values, got {}".format(x.dtype)
#     )
#     # TODO: Change to ValueError
#     assert x.ndim in [1, 2], (
#         "Expected an array of 1 or 2 dimensions, got {}".format(x.ndim)
#     )
#     if x.ndim == 1:
#         x = mx.expand_dims(x, 0)
#
#     if center:
#         x_dim = x.ndim
#         extended_shape = [1] * (3 - x_dim) + list(x.shape)
#



