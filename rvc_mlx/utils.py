from typing import Optional, Union
import mlx.core as mx


def narrow(
    x: mx.array,
    dim: int,
    start: Union[int, mx.array],
    length: int,
) -> mx.array:
    """
    Returns a new tensor that is a narrowed version of tensor `x`. The dimension `dim` is input `x` from start to
    `start + length`. The returned array and input array share the same underlying storage.

    :param x: the tensor to narrow
    :param dim: the dimension along which to narrow
    :param start: index of the element to start the narrowed dimension from. Can be negative, which means indexing from
    the end of `dim`. If `mx.array`, it must be an 0-dimensional integral `mx.array` (bools not allowed)
    :param length: length of the narrowed dimension, must be weakly positive.
    :return: a narrowed version of tensor `x`
    """
    if dim < -x.ndim or dim >= x.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-x.ndim}, {x.ndim-1}], but got {dim})."
        )
    if length < 0:
        raise ValueError("length must be non-negative.")
    if not isinstance(start, mx.array):
        start = mx.array(start)
    if start.item() < 0:
        start = mx.array([x.shape[dim] + start])
    if start.item() + length > x.shape[dim]:
        raise ValueError(
            f"start ({start.item()}) + length ({length}) exceeds dimension size ({x.shape[dim]})."
        )
    slice_size = list(x.shape)
    slice_size[dim] = length
    return mx.slice(x, start_indices=start, axes=(dim,), slice_size=slice_size)


def pad_constant(
    input: mx.array,
    pad: tuple,
    value: Union[int, float],
) -> mx.array:
    if len(pad) % 2 != 0:
        raise ValueError(f"Length of pad must be even but instead it equals {len(pad)}")

    ndim = input.ndim
    num_pad_dims = len(pad) // 2

    if num_pad_dims > ndim:
        raise ValueError(
            f"Length of pad should be no more than twice the number of dimensions of the input. Pad length is {num_pad_dims} while the input has {ndim} dimensions."
        )

    all_pads_non_positive = True

    new_input = input

    # Check for non-positive padding (= narrowing)
    for i in range(ndim - num_pad_dims, ndim):
        pad_idx = 2 * (ndim - i - 1)

        # Left
        if pad[pad_idx] < 0:
            new_input = narrow(
                new_input, dim=i, start=-pad[pad_idx], length=new_input.shape[i] + pad[pad_idx]
            )
        elif pad[pad_idx] != 0:
            all_pads_non_positive = False

        # Right
        if pad[pad_idx + 1] < 0:
            new_input = narrow(
                new_input, dim=i, start=0, length=new_input.shape[i] + pad[pad_idx + 1]
            )
        elif pad[pad_idx + 1] != 0:
            all_pads_non_positive = False

    # If none of the pads are positive, just return the narrowed version
    if all_pads_non_positive:
        return new_input

    new_shape = []
    # Original dimensions
    for i in range(ndim - num_pad_dims):
        new_shape.append(new_input.shape[i])

    start_indices = []
    axes = []
    # Altered dimensions
    for i in range(num_pad_dims):
        pad_idx = len(pad) - ((i + 1) * 2)
        dim_idx = ndim - num_pad_dims + i
        new_dim = input.shape[dim_idx] + pad[pad_idx] + pad[pad_idx + 1]
        new_shape.append(new_dim)
        # If data needs to be copied in, prepare the indices
        if pad[pad_idx] > 0:
            start_indices.append(
                pad[pad_idx] if pad[pad_idx] >= 0 else input.shape[dim_idx] + pad[pad_idx]
            )
            axes.append(dim_idx)
        elif pad[pad_idx + 1] > 0:
            start_indices.append(0)
            axes.append(dim_idx)

    out = mx.full(new_shape, value, dtype=input.dtype)

    # Replace out with new_input
    out = mx.slice_update(out, new_input, start_indices=mx.array(start_indices), axes=axes)

    return out


def sequence_mask(length: mx.array, max_length: Optional[int] = None) -> mx.array:
    """
    Construct a boolean mask of shape (B, max_length) from a (B,) tensor of lengths. The element at index (i, j) is True
    iff j < length[i]. Matches the `sequence_mask` helper used throughout the RVC reference implementation.

    :param length: a 1D `mx.array` of integer lengths.
    :param max_length: the length of the second dimension of the output mask. If None, the maximum value of `length` is
    used.
    :return: a boolean mask of shape (B, max_length).
    """
    if max_length is None:
        max_length = int(length.max().item())
    x = mx.arange(max_length, dtype=length.dtype)
    return mx.expand_dims(x, 0) < mx.expand_dims(length, 1)
