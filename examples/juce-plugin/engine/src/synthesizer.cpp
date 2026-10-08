#include <cmath>
#include <numeric>
#include <sstream>

#include "models.h"

namespace rvc::detail {

namespace {
constexpr float kLreluSlope = 0.1f;
constexpr int kWindowSize = 10;  // relative attention window

std::string idx(const std::string& prefix, int i) { return prefix + "." + std::to_string(i); }

// ---------------------------------------------------------------------------------------------------- text encoder

mx::array relativeEmbeddings(const mx::array& emb, int length) {
    const int pad = std::max(length - (kWindowSize + 1), 0);
    const int start = std::max((kWindowSize + 1) - length, 0);
    mx::array e = pad > 0 ? padAxis(emb, 1, pad, pad) : emb;
    return sliceAxis(e, 1, start, start + 2 * length - 1);                    // (1, 2L-1, K)
}

mx::array relativeToAbsolute(const mx::array& x) {  // (B, H, L, 2L-1) -> (B, H, L, L)
    const int B = x.shape(0), H = x.shape(1), L = x.shape(2);
    mx::array y = mx::reshape(padAxis(x, 3, 0, 1), {B, H, 2 * L * L});
    y = mx::reshape(padAxis(y, 2, 0, L - 1), {B, H, L + 1, 2 * L - 1});
    return mx::slice(y, {0, 0, 0, L - 1}, {B, H, L, 2 * L - 1});
}

mx::array absoluteToRelative(const mx::array& x) {  // (B, H, L, L) -> (B, H, L, 2L-1)
    const int B = x.shape(0), H = x.shape(1), L = x.shape(2);
    mx::array y = mx::reshape(padAxis(x, 3, 0, L - 1), {B, H, L * L + L * (L - 1)});
    y = mx::reshape(padAxis(y, 2, L, 0), {B, H, L, 2 * L});
    return sliceAxis(y, 3, 1, 2 * L);
}

mx::array attention(const Weights& w, const std::string& p, const mx::array& x, const mx::array& attnMask, int nHeads) {
    const int B = x.shape(0), T = x.shape(1), C = x.shape(2), K = C / nHeads;
    auto heads = [&](const mx::array& t) { return mx::transpose(mx::reshape(t, {B, T, nHeads, K}), {0, 2, 1, 3}); };
    mx::array q = heads(conv1d(w, p + ".conv_q", x)) * (1.0f / std::sqrt(static_cast<float>(K)));
    mx::array k = heads(conv1d(w, p + ".conv_k", x));
    mx::array v = heads(conv1d(w, p + ".conv_v", x));

    mx::array scores = mx::matmul(q, mx::transpose(k, {0, 1, 3, 2}));
    mx::array relK = relativeEmbeddings(w.at(p + ".emb_rel_k"), T);
    scores = scores + relativeToAbsolute(mx::matmul(q, mx::expand_dims(mx::transpose(relK, {0, 2, 1}), 0)));
    scores = mx::where(mx::equal(attnMask, mx::array(0.0f)), mx::array(-1e4f, scores.dtype()), scores);
    mx::array probs = mx::astype(mx::softmax(mx::astype(scores, mx::float32), -1), v.dtype());
    mx::array out = mx::matmul(probs, v);
    mx::array relV = relativeEmbeddings(w.at(p + ".emb_rel_v"), T);
    out = out + mx::matmul(absoluteToRelative(probs), mx::expand_dims(relV, 0));
    return conv1d(w, p + ".conv_o", mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, T, C}));
}

mx::array ffn(const Weights& w, const std::string& p, mx::array x, const mx::array& mask, int kernel) {
    const int left = (kernel - 1) / 2, right = kernel / 2;
    auto same = [&](const mx::array& t) { return kernel == 1 ? t : padAxis(t, 1, left, right); };
    x = relu(conv1d(w, p + ".conv_1", same(x * mask)));
    x = conv1d(w, p + ".conv_2", same(x * mask));
    return x * mask;
}

struct Prior {
    mx::array m, logs, mask;
};

Prior textEncoder(const Weights& w, const SynthConfig& cfg, const SynthInputs& in) {
    mx::array x = linear(w, "enc_p.emb_phone", in.phone);
    if (cfg.f0 && in.pitch) x = x + mx::take(w.at("enc_p.emb_pitch.weight"), *in.pitch, 0);
    x = leakyRelu(x * std::sqrt(static_cast<float>(cfg.hiddenChannels)), 0.1f);
    const int T = x.shape(1);
    mx::array mask = mx::astype(mx::less(mx::arange(T), mx::array(in.length)), x.dtype());
    mask = mx::reshape(mask, {1, T, 1});
    mx::array m2 = mx::reshape(mask, {1, 1, T});
    mx::array attnMask = mx::expand_dims(m2, -1) * mx::expand_dims(m2, -2);   // (1, 1, T, T)

    x = x * mask;
    for (int i = 0; i < cfg.nLayers; ++i) {
        x = layerNorm(x + attention(w, idx("enc_p.encoder.attn_layers", i), x, attnMask, cfg.nHeads),
                      w.at(idx("enc_p.encoder.norm_layers_1", i) + ".gamma"),
                      w.at(idx("enc_p.encoder.norm_layers_1", i) + ".beta"));
        x = layerNorm(x + ffn(w, idx("enc_p.encoder.ffn_layers", i), x, mask, cfg.kernelSize),
                      w.at(idx("enc_p.encoder.norm_layers_2", i) + ".gamma"),
                      w.at(idx("enc_p.encoder.norm_layers_2", i) + ".beta"));
    }
    x = x * mask;
    mx::array stats = conv1d(w, "enc_p.proj", x) * mask;
    const int c = cfg.interChannels;
    return {sliceAxis(stats, -1, 0, c), sliceAxis(stats, -1, c, 2 * c), mask};
}

// ------------------------------------------------------------------------------------------------------------ flow

mx::array wavenet(const Weights& w, const std::string& p, mx::array x, const mx::array& mask, const mx::array& g,
                  int hidden, int nLayers) {
    mx::array output = mx::zeros_like(x);
    mx::array cond = conv1d(w, p + ".cond_layer", g);
    for (int i = 0; i < nLayers; ++i) {
        mx::array xin = conv1d(w, idx(p + ".in_layers", i), x, 1, 2, 1);  // kernel 5, dilation 1
        xin = xin + sliceAxis(cond, -1, i * 2 * hidden, (i + 1) * 2 * hidden);
        mx::array acts = mx::tanh(sliceAxis(xin, -1, 0, hidden)) * mx::sigmoid(sliceAxis(xin, -1, hidden, 2 * hidden));
        mx::array rs = conv1d(w, idx(p + ".res_skip_layers", i), acts);
        if (i < nLayers - 1) {
            x = (x + sliceAxis(rs, -1, 0, hidden)) * mask;
            output = output + sliceAxis(rs, -1, hidden, 2 * hidden);
        } else {
            output = output + rs;
        }
    }
    return output * mask;
}

mx::array flowReverse(const Weights& w, const SynthConfig& cfg, mx::array x, const mx::array& mask, const mx::array& g) {
    const int half = cfg.interChannels / 2;
    for (int f = 3; f >= 0; --f) {   // flows.{6,4,2,0}, each preceded (in reverse order) by a channel flip
        x = flipAxis(x, -1);
        const std::string p = idx("flow.flows", 2 * f);
        mx::array x0 = sliceAxis(x, -1, 0, half), x1 = sliceAxis(x, -1, half, 2 * half);
        mx::array h = wavenet(w, p + ".enc", conv1d(w, p + ".pre", x0) * mask, mask, g, cfg.hiddenChannels, 3);
        mx::array m = conv1d(w, p + ".post", h) * mask;
        x = mx::concatenate({x0, (x1 - m) * mask}, -1);
    }
    return x;
}

// --------------------------------------------------------------------------------------------------------- vocoder

mx::array resblock(const Weights& w, const std::string& p, mx::array x, const SynthConfig& cfg, int kernel,
                   const std::vector<int>& dilations) {
    auto pad = [&](int d) { return (kernel * d - d) / 2; };
    if (cfg.resblock == "1") {
        for (size_t j = 0; j < dilations.size(); ++j) {
            const int d = dilations[j];
            mx::array xt = conv1d(w, idx(p + ".convs1", static_cast<int>(j)), leakyRelu(x, kLreluSlope), 1, pad(d), d);
            xt = conv1d(w, idx(p + ".convs2", static_cast<int>(j)), leakyRelu(xt, kLreluSlope), 1, pad(1), 1);
            x = x + xt;
        }
    } else {
        for (size_t j = 0; j < dilations.size(); ++j) {
            const int d = dilations[j];
            x = x + conv1d(w, idx(p + ".convs", static_cast<int>(j)), leakyRelu(x, kLreluSlope), 1, pad(d), d);
        }
    }
    return x;
}

mx::array generator(const Weights& w, const SynthConfig& cfg, mx::array x, const mx::array& g, const SynthInputs& in) {
    std::optional<mx::array> har;
    if (cfg.f0) {
        mx::array sine = sineExcitation(*in.pitchf, cfg.hopLength(), cfg.sr, in.nsfNoise, in.nsfNoiseScale);
        har = mx::tanh(linear(w, "dec.m_source.l_linear", mx::astype(sine, x.dtype())));   // (1, T*upp, 1)
    }
    x = conv1d(w, "dec.conv_pre", x, 1, 3);
    x = x + conv1d(w, "dec.cond", g);
    const int nk = static_cast<int>(cfg.resblockKernelSizes.size());
    for (size_t i = 0; i < cfg.upsampleRates.size(); ++i) {
        const int u = cfg.upsampleRates[i], k = cfg.upsampleKernelSizes[i];
        x = convTranspose1d(w, idx("dec.ups", static_cast<int>(i)), leakyRelu(x, kLreluSlope), u, (k - u) / 2);
        if (har) {
            int stride = 1;
            for (size_t j = i + 1; j < cfg.upsampleRates.size(); ++j) stride *= cfg.upsampleRates[j];
            const bool last = i + 1 == cfg.upsampleRates.size();
            x = x + conv1d(w, idx("dec.noise_convs", static_cast<int>(i)), *har, last ? 1 : stride, last ? 0 : stride / 2);
        }
        std::optional<mx::array> xs;
        for (int j = 0; j < nk; ++j) {
            mx::array r = resblock(w, idx("dec.resblocks", static_cast<int>(i) * nk + j), x, cfg,
                                   cfg.resblockKernelSizes[j], cfg.resblockDilationSizes[j]);
            xs = xs ? *xs + r : r;
        }
        x = *xs * (1.0f / nk);
    }
    x = mx::conv1d(leakyRelu(x, 0.01f), w.at("dec.conv_post.weight"), 1, 3);
    return mx::tanh(x);
}

}  // namespace

// ------------------------------------------------------------------------------------------------------- public API

SynthConfig SynthConfig::fromJson(const Json& j) {
    SynthConfig c;
    c.interChannels = j["inter_channels"].integer();
    c.hiddenChannels = j["hidden_channels"].integer();
    c.filterChannels = j["filter_channels"].integer();
    c.nHeads = j["n_heads"].integer();
    c.nLayers = j["n_layers"].integer();
    c.kernelSize = j["kernel_size"].integer();
    c.resblock = j["resblock"].string();
    c.resblockKernelSizes = j["resblock_kernel_sizes"].ints();
    for (size_t i = 0; i < j["resblock_dilation_sizes"].size(); ++i)
        c.resblockDilationSizes.push_back(j["resblock_dilation_sizes"][i].ints());
    c.upsampleRates = j["upsample_rates"].ints();
    c.upsampleInitialChannel = j["upsample_initial_channel"].integer();
    c.upsampleKernelSizes = j["upsample_kernel_sizes"].ints();
    c.spkEmbedDim = j["spk_embed_dim"].integer();
    c.ginChannels = j["gin_channels"].integer();
    c.sr = j["sr"].integer();
    c.version = j["version"].string();
    c.f0 = j["f0"].boolean();
    return c;
}

int SynthConfig::hopLength() const {
    return std::accumulate(upsampleRates.begin(), upsampleRates.end(), 1, std::multiplies<int>());
}

std::string SynthConfig::architectureKey() const {
    std::ostringstream s;
    auto list = [&](const std::vector<int>& v) { for (int x : v) s << x << ','; s << ';'; };
    s << version << '|' << sr << '|' << f0 << '|' << interChannels << '|' << hiddenChannels << '|' << filterChannels
      << '|' << nHeads << '|' << nLayers << '|' << kernelSize << '|' << resblock << '|' << upsampleInitialChannel
      << '|' << ginChannels << '|';
    list(resblockKernelSizes);
    list(upsampleRates);
    list(upsampleKernelSizes);
    for (const auto& d : resblockDilationSizes) list(d);
    return s.str();
}

mx::array sineExcitation(const mx::array& f0In, int upp, int sr, const std::optional<mx::array>& noiseIn,
                         float noiseScale) {
    const float sineAmp = 0.1f, noiseStd = 0.003f;
    const int B = f0In.shape(0), T = f0In.shape(1);
    mx::array f0 = mx::expand_dims(mx::astype(f0In, mx::float32), -1);                  // (B, T, 1)
    mx::array rad = f0 * (1.0f / sr) * mx::arange(1, upp + 1, mx::float32);            // (B, T, upp)
    mx::array frameEnd = sliceAxis(sliceAxis(rad, 1, 0, T - 1), 2, upp - 1, upp) + 0.5f;
    frameEnd = frameEnd - mx::floor(frameEnd) - 0.5f;                                   // fmod(x + .5, 1) - .5
    mx::array acc = mx::cumsum(frameEnd, 1);
    acc = acc - mx::where(mx::greater_equal(acc, mx::array(0.0f)), mx::floor(acc), mx::ceil(acc));  // fmod(acc, 1)
    rad = rad + padAxis(acc, 1, 1, 0);
    mx::array sines = mx::sin(mx::reshape(rad, {B, T * upp, 1}) * static_cast<float>(2.0 * M_PI)) * sineAmp;
    mx::array uv = mx::repeat(mx::astype(mx::greater(f0, mx::array(0.0f)), mx::float32), upp, 1);
    mx::array noise = noiseIn ? *noiseIn
                              : (noiseScale != 0.0f ? mx::random::normal(sines.shape(), mx::float32)
                                                    : mx::zeros(sines.shape(), mx::float32));
    noise = noise * noiseScale * (uv * noiseStd + (1.0f - uv) * (sineAmp / 3.0f));
    return sines * uv + noise;
}

mx::array synthesize(const Weights& w, const SynthConfig& cfg, const SynthInputs& in) {
    mx::array g = mx::expand_dims(in.speaker, 1);                                       // (1, 1, gin)
    Prior prior = textEncoder(w, cfg, in);
    mx::array eps = in.priorNoise ? *in.priorNoise
                                  : (in.noiseScale != 0.0f ? mx::random::normal(prior.m.shape(), prior.m.dtype())
                                                           : mx::zeros_like(prior.m));
    mx::array zp = (prior.m + mx::exp(prior.logs) * eps * in.noiseScale) * prior.mask;
    mx::array z = flowReverse(w, cfg, zp, prior.mask, g);
    mx::array audio = generator(w, cfg, z * prior.mask, g, in);
    return mx::squeeze(audio, -1);                                                      // (1, T * hop)
}

}  // namespace rvc::detail
