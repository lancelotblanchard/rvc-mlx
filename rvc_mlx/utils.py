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


def pad_reflect_last_dim(input: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """
    Reflect-pad the last dimension of `input` by `pad_left` on the left and `pad_right` on the right, mirroring values
    around the boundary *without* repeating the boundary element. This matches PyTorch's default
    `torch.nn.functional.pad(..., mode="reflect")` and `torch.stft(..., pad_mode="reflect")` behavior.

    Example for a 1D tensor [a, b, c, d, e] with pad_left=2 and pad_right=2: the result is [c, b, a, b, c, d, e, d, c].

    :param input: the tensor to pad. Must have at least 1 dimension.
    :param pad_left: non-negative number of elements to prepend along the last axis. Must be < L (last-dim size).
    :param pad_right: non-negative number of elements to append along the last axis. Must be < L.
    :return: the padded tensor with shape identical to `input` except the last dim is L + pad_left + pad_right.
    """
    if pad_left < 0 or pad_right < 0:
        raise ValueError(f"pad_left and pad_right must be non-negative, got ({pad_left}, {pad_right}).")
    if pad_left == 0 and pad_right == 0:
        return input

    L = input.shape[-1]
    if pad_left >= L or pad_right >= L:
        raise ValueError(
            f"Reflect padding requires pad < last-dim size ({L}); got pad_left={pad_left}, pad_right={pad_right}."
        )

    parts = []
    if pad_left > 0:
        # Mirror indices 1..pad_left in reverse: [pad_left, pad_left-1, ..., 1].
        left_idx = mx.arange(pad_left, 0, -1)
        parts.append(mx.take(input, left_idx, axis=-1))
    parts.append(input)
    if pad_right > 0:
        # Mirror indices L-2, L-3, ..., L-1-pad_right.
        right_idx = mx.arange(L - 2, L - 2 - pad_right, -1)
        parts.append(mx.take(input, right_idx, axis=-1))
    return mx.concatenate(parts, axis=-1)


def interpolate_nearest_axis(x: mx.array, scale_factor: int, axis: int) -> mx.array:
    """
    1D nearest-neighbour upsampling along `axis`, matching
    `torch.nn.functional.interpolate(scale_factor=int, mode="nearest")`.

    For an integer scale factor, PyTorch's nearest mode maps output position `k` to input position `k // scale_factor`,
    which is equivalent to repeating each input element `scale_factor` times. The output length along `axis` is
    `input_length * scale_factor`.

    :param x: input array.
    :param scale_factor: positive integer multiplier for the size along `axis`.
    :param axis: axis to upsample.
    """
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be a positive integer, got {scale_factor}.")
    if scale_factor == 1:
        return x
    return mx.repeat(x, repeats=scale_factor, axis=axis)


def interpolate_linear_axis(x: mx.array, scale_factor: int, axis: int) -> mx.array:
    """
    1D linear interpolation along `axis` with `align_corners=True`, matching
    `torch.nn.functional.interpolate(scale_factor=int, mode="linear", align_corners=True)`.

    The output length along `axis` is `input_length * scale_factor`. With `align_corners=True`, the endpoints of the
    input are exactly preserved at the endpoints of the output. Concretely, for each output position k in
    `[0, N_out - 1]` the float input position is `k * (N_in - 1) / (N_out - 1)`; we linearly blend the floor/ceil
    samples by the fractional part.

    Used by RVC's `SineGen` to upsample the cumulative phase signal to audio rate.

    :param x: input array with length N along `axis`. Must have N >= 2 (so the denominator `N_out - 1` is non-zero).
    :param scale_factor: positive integer multiplier for the size along `axis`.
    :param axis: axis to upsample.
    """
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be a positive integer, got {scale_factor}.")
    if scale_factor == 1:
        return x
    n_in = x.shape[axis]
    if n_in < 2:
        raise ValueError(
            f"interpolate_linear_axis requires input length >= 2 along axis {axis}, got {n_in}."
        )
    n_out = n_in * scale_factor

    # Float input positions for each of the n_out output samples.
    pos = mx.arange(n_out, dtype=mx.float32) * ((n_in - 1) / (n_out - 1))
    pos_low_i = mx.floor(pos).astype(mx.int32)
    # Clamp pos_low + 1 to the last valid index so the rightmost output sample reads the same low/high pair.
    pos_high_i = mx.minimum(pos_low_i + 1, n_in - 1)
    frac = pos - pos_low_i.astype(mx.float32)

    x_low = mx.take(x, pos_low_i, axis=axis)
    x_high = mx.take(x, pos_high_i, axis=axis)

    # Broadcast `frac` so it aligns with `axis` of x_low/x_high. Reshape to (1, ..., 1, n_out, 1, ..., 1).
    shape = [1] * x.ndim
    shape[axis] = n_out
    frac = frac.reshape(shape).astype(x.dtype)

    return x_low * (1 - frac) + x_high * frac


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
