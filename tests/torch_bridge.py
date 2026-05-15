"""
Helpers for testing MLX neural network modules against PyTorch references.

These helpers are tests-only: they live here (rather than in `rvc_mlx/`) so PyTorch stays a dev-only dependency. They
translate PyTorch parameter tensors into MLX arrays, accounting for the channels-first (PyTorch) vs. channels-last (MLX)
convention for 2D convolutions.

The general pattern for a paired-module test is:

    1. Construct the PyTorch reference module, call `.eval()`, and randomize its running BatchNorm statistics so eval
       mode actually uses non-degenerate stats (PyTorch initializes `running_var` to 1 and `running_mean` to 0).
    2. Construct the MLX module with matching hyperparameters.
    3. Use `copy_*` helpers below to mirror parameters across.
    4. Wrap each module in a function that takes a channels-first numpy/tensor input. The MLX wrapper transposes to
       channels-last on the way in and back to channels-first on the way out.
    5. Feed the function pair into `OperationTestSuite`.
"""

from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch


def _to_mx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().numpy())


def copy_conv2d(torch_conv: torch.nn.Conv2d, mlx_conv: nn.Conv2d) -> None:
    """
    Copy weights from PyTorch Conv2d to MLX Conv2d.

    PyTorch Conv2d weight shape: (out_channels, in_channels, kH, kW)
    MLX     Conv2d weight shape: (out_channels, kH, kW, in_channels)  -- channels-last
    """
    w = torch_conv.weight.detach().cpu().numpy().transpose(0, 2, 3, 1)
    mlx_conv.weight = mx.array(w)
    if torch_conv.bias is not None:
        mlx_conv.bias = _to_mx(torch_conv.bias)


def copy_batchnorm(torch_bn: torch.nn.modules.batchnorm._BatchNorm, mlx_bn: nn.BatchNorm) -> None:
    """
    Copy affine params and running statistics from PyTorch BatchNorm to MLX BatchNorm. Both stores have shape
    (num_features,) and identical semantics in eval mode.
    """
    if torch_bn.affine:
        mlx_bn.weight = _to_mx(torch_bn.weight)
        mlx_bn.bias = _to_mx(torch_bn.bias)
    if torch_bn.track_running_stats:
        mlx_bn.running_mean = _to_mx(torch_bn.running_mean)
        mlx_bn.running_var = _to_mx(torch_bn.running_var)


def randomize_bn_stats(torch_module: torch.nn.Module, seed: int = 0) -> None:
    """
    Walk a PyTorch module tree and randomize the running statistics of every BatchNorm. PyTorch initializes them to
    `running_mean=0` and `running_var=1`, which makes eval mode a degenerate no-op for the normalization step. Tests
    want to verify that running stats are honored, so we replace them with non-trivial values.
    """
    rng = np.random.default_rng(seed)
    for m in torch_module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            num_features = m.running_mean.shape[0]
            m.running_mean.copy_(
                torch.from_numpy(rng.standard_normal(num_features).astype(np.float32))
            )
            m.running_var.copy_(
                torch.from_numpy(
                    (rng.standard_normal(num_features).astype(np.float32) ** 2 + 0.1)
                )
            )


def copy_conv_block_res(torch_block, mlx_block) -> None:
    """
    Copy a single `ConvBlockRes` from PyTorch to MLX. The PyTorch reference uses an `nn.Sequential` of length 6
    (Conv, BN, ReLU, Conv, BN, ReLU); the MLX module mirrors this structure with `nn.Sequential` and the same ordering.
    """
    torch_seq = torch_block.conv
    mlx_layers = mlx_block.conv.layers
    copy_conv2d(torch_seq[0], mlx_layers[0])
    copy_batchnorm(torch_seq[1], mlx_layers[1])
    copy_conv2d(torch_seq[3], mlx_layers[3])
    copy_batchnorm(torch_seq[4], mlx_layers[4])
    if hasattr(torch_block, "shortcut"):
        copy_conv2d(torch_block.shortcut, mlx_block.shortcut)


def copy_res_encoder_block(torch_block, mlx_block) -> None:
    """Copy a stack of ConvBlockRes (and the optional pool has no parameters)."""
    for torch_conv_block, mlx_conv_block in zip(torch_block.conv, mlx_block.conv):
        copy_conv_block_res(torch_conv_block, mlx_conv_block)


def to_channels_last(x: mx.array) -> mx.array:
    """Convert a 4D channels-first tensor (B, C, H, W) to MLX channels-last (B, H, W, C)."""
    return mx.transpose(x, (0, 2, 3, 1))


def to_channels_first(x: mx.array) -> mx.array:
    """Convert a 4D channels-last tensor (B, H, W, C) back to (B, C, H, W) for comparison with PyTorch."""
    return mx.transpose(x, (0, 3, 1, 2))


def set_eval(*modules: Iterable) -> None:
    """Put a heterogeneous set of MLX and PyTorch modules into eval mode."""
    for m in modules:
        if isinstance(m, torch.nn.Module):
            m.eval()
        else:
            # MLX nn.Module exposes either .eval() or .train(False); cover both.
            if hasattr(m, "eval"):
                m.eval()
            else:
                m.train(False)
