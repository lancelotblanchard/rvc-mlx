#include <cmath>

#include "models.h"

namespace rvc::detail {

Hubert::Hubert(Weights weights) : w_(std::move(weights)) {
    Json cfg = Json::parse(w_.metaOr("config", "{}"));
    if (cfg.has("conv_layers")) {
        const Json& layers = cfg["conv_layers"];
        for (size_t i = 0; i < layers.size(); ++i)
            convLayers_.push_back({layers[i][0].integer(), layers[i][1].integer(), layers[i][2].integer()});
        numHeads_ = cfg["num_heads"].integer();
        numLayers_ = cfg["num_layers"].integer();
        convPos_ = cfg["conv_pos"].integer();
        convPosGroups_ = cfg["conv_pos_groups"].integer();
    } else {
        convLayers_ = {{512, 10, 5}, {512, 3, 2}, {512, 3, 2}, {512, 3, 2}, {512, 3, 2}, {512, 2, 2}, {512, 2, 2}};
    }
    hasFinalProj_ = w_.metaOr("has_final_proj", "1") == "1";
    dtype_ = w_.at("post_extract_proj.weight").dtype();
}

mx::array Hubert::extract(const mx::array& audio, int outputLayer) const {
    // Feature extractor: strided convs (no bias); the first has a per-channel GroupNorm (computed in fp32).
    mx::array x = mx::expand_dims(mx::astype(audio, dtype_), -1);  // (1, L, 1)
    for (size_t i = 0; i < convLayers_.size(); ++i) {
        const std::string p = "feature_extractor.conv_layers." + std::to_string(i);
        x = mx::conv1d(x, w_.at(p + ".conv.weight"), convLayers_[i][2]);
        if (i == 0) {
            mx::array xf = mx::astype(x, mx::float32);
            mx::array mean = mx::mean(xf, 1, true);
            mx::array var = mx::var(xf, 1, true);
            xf = (xf - mean) * mx::rsqrt(var + 1e-5f);
            x = mx::astype(xf * w_.at(p + ".norm.weight") + w_.at(p + ".norm.bias"), dtype_);
        }
        x = gelu(x);
    }
    x = linear(w_, "post_extract_proj", layerNorm(w_, "layer_norm", x));

    // Positional convolution (grouped, weight-norm folded) with SamePad, then post-LN transformer layers.
    mx::array pos = conv1d(w_, "encoder.pos_conv", x, 1, convPos_ / 2, 1, convPosGroups_);
    if (convPos_ % 2 == 0) pos = sliceAxis(pos, 1, 0, pos.shape(1) - 1);
    x = layerNorm(w_, "encoder.layer_norm", x + gelu(pos));

    const int B = x.shape(0), T = x.shape(1), C = x.shape(2), H = numHeads_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(C / H));
    const int n = std::min(outputLayer, numLayers_);
    for (int i = 0; i < n; ++i) {
        const std::string p = "encoder.layers." + std::to_string(i);
        auto heads = [&](const mx::array& t) { return mx::transpose(mx::reshape(t, {B, T, H, C / H}), {0, 2, 1, 3}); };
        mx::array q = heads(linear(w_, p + ".self_attn.q_proj", x));
        mx::array k = heads(linear(w_, p + ".self_attn.k_proj", x));
        mx::array v = heads(linear(w_, p + ".self_attn.v_proj", x));
        mx::array o = mx::fast::scaled_dot_product_attention(q, k, v, scale);
        o = mx::reshape(mx::transpose(o, {0, 2, 1, 3}), {B, T, C});
        x = layerNorm(w_, p + ".self_attn_layer_norm", x + linear(w_, p + ".self_attn.out_proj", o));
        mx::array ff = linear(w_, p + ".fc2", gelu(linear(w_, p + ".fc1", x)));
        x = layerNorm(w_, p + ".final_layer_norm", x + ff);
    }
    return x;
}

mx::array Hubert::features(const mx::array& audio, const std::string& version) const {
    if (version == "v1") {
        if (!hasFinalProj_) throw std::runtime_error("This content encoder has no final_proj; it can't drive v1 voices");
        return linear(w_, "final_proj", extract(audio, 9));
    }
    return extract(audio, 12);
}

}  // namespace rvc::detail
