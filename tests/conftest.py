"""
Shared pytest configuration.

MLX's Linux CPU backend JIT-compiles fused kernels (`mx.compile`, used internally by `mlx.nn` activations) with the
host C++ compiler. Some toolchains (e.g. GCC 13 on Ubuntu 24.04) reject the generated preamble, which aborts the
process. Graph compilation is a pure optimisation, so we disable it on Linux to keep the suite portable. macOS (Metal)
is unaffected and keeps compilation on.
"""

import sys

import mlx.core as mx

if sys.platform.startswith("linux"):
    mx.disable_compile()
