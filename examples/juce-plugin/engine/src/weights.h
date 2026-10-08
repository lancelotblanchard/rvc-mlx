// Converted-model loading and small functional NN helpers shared by the model implementations.
#pragma once

#include <mlx/mlx.h>

#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>

namespace rvc::detail {

namespace mx = mlx::core;

struct Weights {
    std::unordered_map<std::string, mx::array> arrays;
    std::unordered_map<std::string, std::string> meta;
    std::string path;

    bool has(const std::string& key) const { return arrays.count(key) > 0; }
    const mx::array& at(const std::string& key) const;
    std::string metaOr(const std::string& key, const std::string& fallback) const {
        auto it = meta.find(key);
        return it == meta.end() ? fallback : it->second;
    }
};

/// Loads a converted file, checks its `kind`, casts floating tensors to `dtype` and materialises them.
Weights loadWeights(const std::string& path, const std::string& expectedKind, mx::Dtype dtype);

// ----------------------------------------------------------------------------------------------- functional layers

inline mx::array linear(const Weights& w, const std::string& name, const mx::array& x) {
    mx::array y = mx::matmul(x, mx::transpose(w.at(name + ".weight")));
    return w.has(name + ".bias") ? y + w.at(name + ".bias") : y;
}

/// Channels-last 1-D convolution: x (B, T, Cin), weight (Cout, K, Cin / groups).
inline mx::array conv1d(const Weights& w, const std::string& name, const mx::array& x, int stride = 1,
                        int padding = 0, int dilation = 1, int groups = 1) {
    mx::array y = mx::conv1d(x, w.at(name + ".weight"), stride, padding, dilation, groups);
    return w.has(name + ".bias") ? y + w.at(name + ".bias") : y;
}

/// PyTorch-semantics transposed convolution: x (B, T, Cin), weight (Cout, K, Cin).
inline mx::array convTranspose1d(const Weights& w, const std::string& name, const mx::array& x, int stride,
                                 int padding) {
    mx::array y = mx::conv_transpose1d(x, w.at(name + ".weight"), stride, padding);
    return w.has(name + ".bias") ? y + w.at(name + ".bias") : y;
}

inline mx::array conv2d(const Weights& w, const std::string& name, const mx::array& x, int padding) {
    mx::array y = mx::conv2d(x, w.at(name + ".weight"), {1, 1}, {padding, padding});
    return w.has(name + ".bias") ? y + w.at(name + ".bias") : y;
}

inline mx::array layerNorm(const mx::array& x, const mx::array& gamma, const mx::array& beta, float eps = 1e-5f) {
    return mx::fast::layer_norm(x, gamma, beta, eps);
}

inline mx::array layerNorm(const Weights& w, const std::string& name, const mx::array& x, float eps = 1e-5f) {
    return layerNorm(x, w.at(name + ".weight"), w.at(name + ".bias"), eps);
}

/// Inference-mode BatchNorm (running statistics), channels-last.
inline mx::array batchNorm(const Weights& w, const std::string& name, const mx::array& x, float eps = 1e-5f) {
    mx::array scale = w.at(name + ".weight") * mx::rsqrt(w.at(name + ".running_var") + eps);
    return (x - w.at(name + ".running_mean")) * scale + w.at(name + ".bias");
}

inline mx::array gelu(const mx::array& x) {
    return x * 0.5f * (1.0f + mx::erf(x * static_cast<float>(M_SQRT1_2)));
}

inline mx::array leakyRelu(const mx::array& x, float slope) { return mx::maximum(x, x * slope); }

inline mx::array relu(const mx::array& x) { return mx::maximum(x, mx::zeros_like(x)); }

/// x[..., start:stop] along `axis` (stride 1).
inline mx::array sliceAxis(const mx::array& x, int axis, int start, int stop) {
    if (axis < 0) axis += x.ndim();
    mx::Shape lo(x.ndim(), 0), hi(x.shape().begin(), x.shape().end());
    lo[axis] = start;
    hi[axis] = stop;
    return mx::slice(x, lo, hi);
}

inline mx::array padAxis(const mx::array& x, int axis, int before, int after) {
    if (axis < 0) axis += x.ndim();
    std::vector<std::pair<int, int>> widths(x.ndim(), {0, 0});
    widths[axis] = {before, after};
    return mx::pad(x, widths);
}

/// Reverse along `axis`.
inline mx::array flipAxis(const mx::array& x, int axis) {
    if (axis < 0) axis += x.ndim();
    int n = x.shape(axis);
    return mx::take(x, mx::arange(n - 1, -1, -1), axis);
}

std::string format(const mx::Shape& shape);

}  // namespace rvc::detail
