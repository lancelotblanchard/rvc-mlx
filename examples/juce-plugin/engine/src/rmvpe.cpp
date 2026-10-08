#include <algorithm>
#include <cmath>

#include "models.h"

namespace rvc::detail {

namespace {
constexpr int kNfft = 1024, kHop = 160, kBins = 360, kPad = 4;
constexpr double kCentsOffset = 1997.3794084376191, kCentsStep = 20.0;

/// numpy-style reflect padding of the last axis of a (1, L) array (works for pads longer than L too).
mx::array reflectPad(const mx::array& x, int left, int right) {
    const int n = x.shape(-1);
    std::vector<int32_t> idx(static_cast<size_t>(n + left + right));
    const int period = std::max(1, 2 * (n - 1));
    for (int i = -left; i < n + right; ++i) {
        int j = n == 1 ? 0 : std::abs(i) % period;
        if (j >= n) j = period - j;
        idx[static_cast<size_t>(i + left)] = j;
    }
    return mx::take(x, mx::array(idx.data(), {static_cast<int>(idx.size())}, mx::int32), -1);
}
}  // namespace

Rmvpe::Rmvpe(Weights weights) : w_(std::move(weights)) {
    Json cfg = Json::parse(w_.metaOr("config", "{}"));
    if (cfg.has("n_blocks")) {
        nBlocks_ = cfg["n_blocks"].integer();
        nGru_ = cfg["n_gru"].integer();
        enDeLayers_ = cfg["en_de_layers"].integer();
        interLayers_ = cfg["inter_layers"].integer();
        enOutChannels_ = cfg["en_out_channels"].integer();
    }
    if (!w_.has("mel_basis"))
        throw std::runtime_error("'" + w_.path + "' has no mel_basis; re-convert it with rvc-mlx >= 0.2");
    std::vector<float> win(kNfft);
    for (int n = 0; n < kNfft; ++n) win[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * n / kNfft));
    window_ = mx::array(win.data(), {kNfft}, mx::float32);
}

mx::array Rmvpe::mel(const mx::array& audio) const {
    mx::array x = reflectPad(mx::astype(audio, mx::float32), kNfft / 2, kNfft / 2);
    mx::eval(x);
    const int length = x.shape(-1);
    const int frames = 1 + (length - kNfft) / kHop;
    mx::array f = mx::as_strided(x, {1, frames, kNfft}, {length, kHop, 1}, 0) * window_;
    mx::array mag = mx::abs(mx::fft::rfft(f, -1));                             // (1, frames, 513)
    mx::array melSpec = mx::matmul(mx::astype(w_.at("mel_basis"), mx::float32), mx::transpose(mag, {0, 2, 1}));
    return mx::log(mx::maximum(melSpec, mx::array(1e-5f)));                  // (1, 128, frames)
}

mx::array Rmvpe::convBlockRes(const std::string& p, const mx::array& x, int inC, int outC) const {
    mx::array y = relu(batchNorm(w_, p + ".conv.layers.1", conv2d(w_, p + ".conv.layers.0", x, 1)));
    y = relu(batchNorm(w_, p + ".conv.layers.4", conv2d(w_, p + ".conv.layers.3", y, 1)));
    return y + (inC != outC ? conv2d(w_, p + ".shortcut", x, 0) : x);
}

mx::array Rmvpe::biGru(const mx::array& x) const {
    // Forward and backward directions run as one batched recurrence: state (2, 1, H).
    mx::array h_in = x;
    for (int layer = 0; layer < nGru_; ++layer) {
        const std::string f = "gru.forward_grus." + std::to_string(layer);
        const std::string b = "gru.backward_grus." + std::to_string(layer);
        const int T = h_in.shape(1);
        const int H = w_.at(f + ".Wh").shape(1);
        mx::array xf = mx::matmul(h_in, mx::transpose(w_.at(f + ".Wx"))) + w_.at(f + ".b");
        mx::array xb = mx::matmul(flipAxis(h_in, 1), mx::transpose(w_.at(b + ".Wx"))) + w_.at(b + ".b");
        mx::array X = mx::stack({xf, xb}, 0);                                     // (2, 1, T, 3H)
        mx::array Wh = mx::stack({mx::transpose(w_.at(f + ".Wh")), mx::transpose(w_.at(b + ".Wh"))}, 0);  // (2, H, 3H)
        mx::array bhn = mx::reshape(mx::stack({w_.at(f + ".bhn"), w_.at(b + ".bhn")}, 0), {2, 1, H});
        mx::array h = mx::zeros({2, 1, H}, X.dtype());
        std::vector<mx::array> outs;
        outs.reserve(static_cast<size_t>(T));
        for (int t = 0; t < T; ++t) {
            mx::array xt = mx::squeeze(sliceAxis(X, 2, t, t + 1), 2);           // (2, 1, 3H)
            mx::array hp = mx::matmul(h, Wh);                                    // (2, 1, 3H)
            mx::array rz = mx::sigmoid(sliceAxis(xt, -1, 0, 2 * H) + sliceAxis(hp, -1, 0, 2 * H));
            mx::array r = sliceAxis(rz, -1, 0, H), z = sliceAxis(rz, -1, H, 2 * H);
            mx::array n = mx::tanh(sliceAxis(xt, -1, 2 * H, 3 * H) + r * (sliceAxis(hp, -1, 2 * H, 3 * H) + bhn));
            h = (1.0f - z) * n + z * h;
            outs.push_back(h);
            if ((t + 1) % 512 == 0) mx::eval(h);  // keep the lazy graph bounded on long inputs
        }
        mx::array seq = mx::stack(outs, 2);                                      // (2, 1, T, H)
        mx::array fwd = mx::squeeze(sliceAxis(seq, 0, 0, 1), 0);
        mx::array bwd = flipAxis(mx::squeeze(sliceAxis(seq, 0, 1, 2), 0), 1);
        h_in = mx::concatenate({fwd, bwd}, -1);
    }
    return h_in;
}

mx::array Rmvpe::salience(const mx::array& melIn) const {
    const int frames = melIn.shape(-1);
    const int padded = 32 * ((frames - 1) / 32 + 1);
    mx::array m = padAxis(melIn, -1, 0, padded - frames);
    mx::array x = mx::expand_dims(mx::transpose(m, {0, 2, 1}), -1);          // (1, T, 128, 1)

    // U-Net encoder.
    x = batchNorm(w_, "unet.encoder.bn", x);
    std::vector<mx::array> skips;
    int inC = 1, outC = enOutChannels_;
    for (int i = 0; i < enDeLayers_; ++i) {
        for (int j = 0; j < nBlocks_; ++j) {
            const std::string p = "unet.encoder.layers." + std::to_string(i) + ".conv." + std::to_string(j);
            x = convBlockRes(p, x, j == 0 ? inC : outC, outC);
        }
        skips.push_back(x);
        const int B = x.shape(0), H = x.shape(1), W = x.shape(2), C = x.shape(3);
        x = mx::mean(mx::reshape(x, {B, H / 2, 2, W / 2, 2, C}), std::vector<int>{2, 4});  // AvgPool2d(2)
        inC = outC;
        outC *= 2;
    }
    // Intermediate.
    const int top = outC;  // encoder.out_channel
    for (int i = 0; i < interLayers_; ++i) {
        const int cin = i == 0 ? top / 2 : top;
        for (int j = 0; j < nBlocks_; ++j) {
            const std::string p = "unet.intermediate.layers." + std::to_string(i) + ".conv." + std::to_string(j);
            x = convBlockRes(p, x, j == 0 ? cin : top, top);
        }
    }
    // Decoder.
    int c = top;
    for (int i = 0; i < enDeLayers_; ++i) {
        const std::string p = "unet.decoder.layers." + std::to_string(i);
        const int out = c / 2;
        x = mx::conv_transpose2d(x, w_.at(p + ".conv1.layers.0.weight"), {2, 2}, {1, 1}, {1, 1}, {1, 1});
        x = relu(batchNorm(w_, p + ".conv1.layers.1", x));
        x = mx::concatenate({x, skips[skips.size() - 1 - i]}, -1);
        for (int j = 0; j < nBlocks_; ++j)
            x = convBlockRes(p + ".conv2." + std::to_string(j), x, j == 0 ? out * 2 : out, out);
        c = out;
    }
    x = conv2d(w_, "cnn", x, 1);                                               // (1, T, 128, 3)
    x = mx::transpose(x, {0, 1, 3, 2});
    x = mx::reshape(x, {x.shape(0), x.shape(1), -1});                           // (1, T, 384)
    x = mx::sigmoid(linear(w_, "linear", biGru(x)));                            // (1, T, 360)
    return sliceAxis(x, 1, 0, frames);
}

std::vector<double> Rmvpe::f0(const std::vector<float>& audio16k, float threshold) const {
    mx::array audio(audio16k.data(), {1, static_cast<int>(audio16k.size())}, mx::float32);
    mx::array s = mx::astype(salience(mel(audio)), mx::float32);
    mx::eval(s);
    const int T = s.shape(1);
    const float* sal = s.data<float>();
    std::vector<double> f0(static_cast<size_t>(T), 0.0);
    for (int t = 0; t < T; ++t) {
        const float* row = sal + static_cast<size_t>(t) * kBins;
        const int center = static_cast<int>(std::max_element(row, row + kBins) - row);
        const float peak = row[center];
        double num = 0, den = 0;
        for (int k = center - kPad; k <= center + kPad; ++k) {
            if (k < 0 || k >= kBins) continue;  // zero padding in the reference
            num += row[k] * (kCentsStep * k + kCentsOffset);
            den += row[k];
        }
        if (peak <= threshold || den == 0) continue;
        const double hz = 10.0 * std::pow(2.0, (num / den) / 1200.0);
        f0[static_cast<size_t>(t)] = hz == 10.0 ? 0.0 : hz;
    }
    return f0;
}

}  // namespace rvc::detail
