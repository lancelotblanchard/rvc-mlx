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


def copy_conv_transpose2d(
    torch_conv: torch.nn.ConvTranspose2d, mlx_conv: nn.ConvTranspose2d
) -> None:
    """
    Copy weights from PyTorch ConvTranspose2d to MLX ConvTranspose2d.

    PyTorch ConvTranspose2d weight shape: (in_channels, out_channels, kH, kW)
    MLX     ConvTranspose2d weight shape: (out_channels, kH, kW, in_channels)
    """
    w = torch_conv.weight.detach().cpu().numpy().transpose(1, 2, 3, 0)
    mlx_conv.weight = mx.array(w)
    if torch_conv.bias is not None:
        mlx_conv.bias = _to_mx(torch_conv.bias)


def copy_res_decoder_block(torch_block, mlx_block) -> None:
    """
    Copy a `ResDecoderBlock` (ConvTranspose + BN + ReLU, then a stack of ConvBlockRes). Both implementations wrap the
    transpose-convolution stage in an `nn.Sequential` of length 3.
    """
    torch_seq = torch_block.conv1
    mlx_layers = mlx_block.conv1.layers
    copy_conv_transpose2d(torch_seq[0], mlx_layers[0])
    copy_batchnorm(torch_seq[1], mlx_layers[1])
    for torch_conv_block, mlx_conv_block in zip(torch_block.conv2, mlx_block.conv2):
        copy_conv_block_res(torch_conv_block, mlx_conv_block)


def copy_encoder(torch_encoder, mlx_encoder) -> None:
    """Copy the U-Net Encoder: top-level BatchNorm + a list of ResEncoderBlocks."""
    copy_batchnorm(torch_encoder.bn, mlx_encoder.bn)
    for torch_layer, mlx_layer in zip(torch_encoder.layers, mlx_encoder.layers):
        copy_res_encoder_block(torch_layer, mlx_layer)


def copy_intermediate(torch_inter, mlx_inter) -> None:
    """Copy the U-Net Intermediate: a list of ResEncoderBlocks with kernel_size=None."""
    for torch_layer, mlx_layer in zip(torch_inter.layers, mlx_inter.layers):
        copy_res_encoder_block(torch_layer, mlx_layer)


def copy_decoder(torch_decoder, mlx_decoder) -> None:
    """Copy the U-Net Decoder: a list of ResDecoderBlocks."""
    for torch_layer, mlx_layer in zip(torch_decoder.layers, mlx_decoder.layers):
        copy_res_decoder_block(torch_layer, mlx_layer)


def copy_deep_unet(torch_unet, mlx_unet) -> None:
    """Copy the full DeepUnet (Encoder + Intermediate + Decoder)."""
    copy_encoder(torch_unet.encoder, mlx_unet.encoder)
    copy_intermediate(torch_unet.intermediate, mlx_unet.intermediate)
    copy_decoder(torch_unet.decoder, mlx_unet.decoder)


def _copy_single_gru(
    t_weight_ih: torch.Tensor,
    t_weight_hh: torch.Tensor,
    t_bias_ih: torch.Tensor,
    t_bias_hh: torch.Tensor,
    mlx_gru: nn.GRU,
) -> None:
    """
    Copy the four tensors of one direction/layer of a PyTorch GRU into one MLX `nn.GRU`.

    PyTorch parameter layouts (all gates in the order r, z, n):
        weight_ih: (3 * hidden, input)
        weight_hh: (3 * hidden, hidden)
        bias_ih:   (3 * hidden,)
        bias_hh:   (3 * hidden,)

    MLX `nn.GRU` parameter layouts (also r, z, n):
        Wx: (3 * hidden, input)    -- same as weight_ih
        Wh: (3 * hidden, hidden)   -- same as weight_hh
        b:  (3 * hidden,)          -- the r/z slots get bias_ih + bias_hh; the n slot gets bias_ih only
        bhn:(hidden,)              -- the n slot's hidden bias, taken from bias_hh[2H:3H]

    The split is necessary because MLX applies `bhn` inside the reset-gate-gated path:
    `n = tanh(W_xn x + b_n + r * (W_hn h + bhn))`, matching PyTorch's
    `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`.
    """
    H = mlx_gru.hidden_size
    weight_ih = t_weight_ih.detach().cpu().numpy()
    weight_hh = t_weight_hh.detach().cpu().numpy()
    bias_ih = t_bias_ih.detach().cpu().numpy()
    bias_hh = t_bias_hh.detach().cpu().numpy()

    mlx_gru.Wx = mx.array(weight_ih)
    mlx_gru.Wh = mx.array(weight_hh)

    combined_b = bias_ih.copy()
    combined_b[: 2 * H] = bias_ih[: 2 * H] + bias_hh[: 2 * H]  # r and z gates
    # n gate's b stays as bias_ih[2H:3H] (already in combined_b from the .copy()).
    mlx_gru.b = mx.array(combined_b)
    mlx_gru.bhn = mx.array(bias_hh[2 * H : 3 * H])


def copy_bi_gru(torch_gru: torch.nn.GRU, mlx_bigru) -> None:
    """
    Copy a PyTorch bidirectional, multi-layer `nn.GRU` into our `BiGRU` (which is built from MLX's single-direction
    primitives). For layer i, PyTorch exposes:
        weight_ih_l{i}, weight_hh_l{i}, bias_ih_l{i}, bias_hh_l{i}                     -- forward
        weight_ih_l{i}_reverse, weight_hh_l{i}_reverse, bias_ih_l{i}_reverse, bias_hh_l{i}_reverse  -- backward
    """
    assert torch_gru.bidirectional, "copy_bi_gru expects a bidirectional torch GRU"
    for i, (fgru, bgru) in enumerate(zip(mlx_bigru.forward_grus, mlx_bigru.backward_grus)):
        _copy_single_gru(
            getattr(torch_gru, f"weight_ih_l{i}"),
            getattr(torch_gru, f"weight_hh_l{i}"),
            getattr(torch_gru, f"bias_ih_l{i}"),
            getattr(torch_gru, f"bias_hh_l{i}"),
            fgru,
        )
        _copy_single_gru(
            getattr(torch_gru, f"weight_ih_l{i}_reverse"),
            getattr(torch_gru, f"weight_hh_l{i}_reverse"),
            getattr(torch_gru, f"bias_ih_l{i}_reverse"),
            getattr(torch_gru, f"bias_hh_l{i}_reverse"),
            bgru,
        )


def copy_linear(torch_linear: torch.nn.Linear, mlx_linear: nn.Linear) -> None:
    """PyTorch and MLX Linear share weight shape (out, in) and bias shape (out,)."""
    mlx_linear.weight = _to_mx(torch_linear.weight)
    if torch_linear.bias is not None:
        mlx_linear.bias = _to_mx(torch_linear.bias)


def copy_e2e(torch_e2e, mlx_e2e) -> None:
    """Copy the full E2E pitch network (DeepUnet + Conv + BiGRU + Linear)."""
    copy_deep_unet(torch_e2e.unet, mlx_e2e.unet)
    copy_conv2d(torch_e2e.cnn, mlx_e2e.cnn)
    # The reference wraps BiGRU + Linear + Dropout + Sigmoid in a single `fc = nn.Sequential(...)`.
    torch_fc = torch_e2e.fc
    copy_bi_gru(torch_fc[0].gru, mlx_e2e.gru)
    copy_linear(torch_fc[1], mlx_e2e.linear)


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
