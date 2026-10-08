#include "dsp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "rvc/rvc.h"

namespace rvc::detail {

namespace {
// scipy.signal.butter(N=5, Wn=48, btype="high", fs=16000) and scipy.signal.lfilter_zi(b, a).
constexpr std::array<double, 6> kB = {0.9699606451838447, -4.849803225919223, 9.699606451838447,
                                      -9.699606451838447, 4.849803225919223,  -0.9699606451838447};
constexpr std::array<double, 6> kA = {1.0, -4.939001819168364, 9.757863526739543,
                                      -9.639544849413458, 4.761506797356209, -0.9408236532054606};
constexpr std::array<double, 5> kZi = {-0.9699607413707367, 3.879842959615721, -5.81976443080129,
                                       3.879842948235015, -0.9699607356787477};

std::vector<double> lfilter(const std::vector<double>& x, double initScale) {
    std::array<double, 5> z{};
    for (size_t i = 0; i < z.size(); ++i) z[i] = kZi[i] * initScale;
    std::vector<double> y(x.size());
    for (size_t n = 0; n < x.size(); ++n) {
        const double xn = x[n];
        const double yn = kB[0] * xn + z[0];
        for (size_t i = 0; i < 4; ++i) z[i] = kB[i + 1] * xn + z[i + 1] - kA[i + 1] * yn;
        z[4] = kB[5] * xn - kA[5] * yn;
        y[n] = yn;
    }
    return y;
}
}  // namespace

std::vector<double> highpass(const std::vector<double>& x) {
    constexpr int padlen = 3 * 6;
    const size_t n = x.size();
    if (n <= static_cast<size_t>(padlen)) throw std::invalid_argument("audio is too short to filter");
    std::vector<double> ext;
    ext.reserve(n + 2 * padlen);
    for (int i = padlen; i >= 1; --i) ext.push_back(2 * x[0] - x[static_cast<size_t>(i)]);
    ext.insert(ext.end(), x.begin(), x.end());
    for (int i = 2; i <= padlen + 1; ++i) ext.push_back(2 * x[n - 1] - x[n - static_cast<size_t>(i)]);

    std::vector<double> y = lfilter(ext, ext.front());
    std::reverse(y.begin(), y.end());
    y = lfilter(y, y.front());
    std::reverse(y.begin(), y.end());
    return std::vector<double>(y.begin() + padlen, y.end() - padlen);
}

std::vector<double> reflectPad(const std::vector<double>& x, int left, int right) {
    const long n = static_cast<long>(x.size());
    std::vector<double> out(static_cast<size_t>(n + left + right));
    const long period = std::max<long>(1, 2 * (n - 1));
    for (long i = -left; i < n + right; ++i) {
        long j = n == 1 ? 0 : std::labs(i) % period;
        if (j >= n) j = period - j;
        out[static_cast<size_t>(i + left)] = x[static_cast<size_t>(j)];
    }
    return out;
}

std::vector<double> frameRms(const std::vector<double>& y, int frameLength, int hopLength) {
    const size_t pad = static_cast<size_t>(frameLength / 2);
    const size_t len = y.size() + 2 * pad;
    if (len < static_cast<size_t>(frameLength)) return {0.0};
    // Prefix sums of squares over the zero-padded signal.
    std::vector<double> prefix(len + 1, 0.0);
    for (size_t i = 0; i < len; ++i) {
        const double v = (i >= pad && i - pad < y.size()) ? y[i - pad] : 0.0;
        prefix[i + 1] = prefix[i] + v * v;
    }
    const size_t frames = 1 + (len - static_cast<size_t>(frameLength)) / static_cast<size_t>(hopLength);
    std::vector<double> out(frames);
    for (size_t f = 0; f < frames; ++f) {
        const size_t s = f * static_cast<size_t>(hopLength);
        out[f] = std::sqrt(std::max(0.0, prefix[s + static_cast<size_t>(frameLength)] - prefix[s]) / frameLength);
    }
    return out;
}

std::vector<double> interpLinear(const std::vector<double>& x, size_t size) {
    const double n = static_cast<double>(x.size());
    std::vector<double> out(size);
    for (size_t i = 0; i < size; ++i) {
        double pos = (static_cast<double>(i) + 0.5) * (n / static_cast<double>(size)) - 0.5;
        pos = std::clamp(pos, 0.0, n - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = std::min(lo + 1, x.size() - 1);
        const double frac = pos - static_cast<double>(lo);
        out[i] = x[lo] * (1 - frac) + x[hi] * frac;
    }
    return out;
}

void changeRms(const std::vector<double>& source, int sourceRate, std::vector<float>& target, int targetRate,
               double rate) {
    std::vector<double> t(target.begin(), target.end());
    std::vector<double> rms1 = interpLinear(frameRms(source, sourceRate / 2 * 2, sourceRate / 2), target.size());
    std::vector<double> rms2 = interpLinear(frameRms(t, targetRate / 2 * 2, targetRate / 2), target.size());
    for (size_t i = 0; i < target.size(); ++i) {
        const double r2 = std::max(rms2[i], 1e-6);
        target[i] = static_cast<float>(t[i] * std::pow(rms1[i], 1 - rate) * std::pow(r2, rate - 1));
    }
}

std::vector<int> coarseF0(const std::vector<double>& f0) {
    const double melMin = 1127 * std::log(1 + 50.0 / 700), melMax = 1127 * std::log(1 + 1100.0 / 700);
    std::vector<int> out(f0.size());
    for (size_t i = 0; i < f0.size(); ++i) {
        double mel = 1127 * std::log(1 + f0[i] / 700);
        if (mel > 0) mel = (mel - melMin) * 254 / (melMax - melMin) + 1;
        mel = std::clamp(mel, 1.0, 255.0);
        out[i] = static_cast<int>(std::nearbyint(mel));  // round-half-even, like np.rint
    }
    return out;
}

std::vector<long> splitPoints(const std::vector<double>& audio, int window, long tMax, long tCenter, long tQuery) {
    std::vector<long> ts;
    std::vector<double> pad = reflectPad(audio, window / 2, window / 2);
    if (static_cast<long>(pad.size()) <= tMax) return ts;
    const long n = static_cast<long>(audio.size());
    std::vector<double> sum(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < window; ++i)  // same accumulation order as the reference (exact ties matter)
        for (long j = 0; j < n; ++j) sum[static_cast<size_t>(j)] += std::fabs(pad[static_cast<size_t>(i + j)]);
    for (long t = tCenter; t < n; t += tCenter) {
        const long lo = t - tQuery, hi = std::min(n, t + tQuery);
        long best = lo;
        for (long j = lo; j < hi; ++j)
            if (sum[static_cast<size_t>(j)] < sum[static_cast<size_t>(best)]) best = j;
        ts.push_back(best);
    }
    return ts;
}

}  // namespace rvc::detail

// ---------------------------------------------------------------------------------------------------- resampling

namespace rvc {

namespace {
double besselI0(double x) {
    double sum = 1, term = 1;
    for (int k = 1; k < 50; ++k) {
        term *= (x / (2 * k)) * (x / (2 * k));
        sum += term;
        if (term < 1e-12 * sum) break;
    }
    return sum;
}
}  // namespace

std::vector<float> resample(const std::vector<float>& input, double fromRate, double toRate) {
    if (fromRate <= 0 || toRate <= 0) throw std::invalid_argument("sample rates must be positive");
    if (fromRate == toRate || input.empty()) return input;

    constexpr int kZeroCrossings = 32;
    constexpr int kTableRes = 512;  // table samples per zero crossing (linear interpolation in between)
    constexpr double kBeta = 9.0;   // Kaiser: ~90 dB stop band
    const double ratio = toRate / fromRate;
    const double cutoff = std::min(1.0, ratio) * 0.95;  // keep a transition band below Nyquist

    // h(u) = cutoff * sinc(cutoff * u) * kaiser(u / span), tabulated over u in [0, span] (input samples).
    const double span = kZeroCrossings / cutoff;
    const int tableSize = static_cast<int>(std::ceil(span * kTableRes)) + 2;
    std::vector<double> table(static_cast<size_t>(tableSize));
    const double i0beta = besselI0(kBeta);
    for (int i = 0; i < tableSize; ++i) {
        const double u = static_cast<double>(i) / kTableRes;
        const double r = u / span;
        const double win = r >= 1 ? 0.0 : besselI0(kBeta * std::sqrt(1 - r * r)) / i0beta;
        const double x = M_PI * cutoff * u;
        table[static_cast<size_t>(i)] = cutoff * (u == 0 ? 1.0 : std::sin(x) / x) * win;
    }
    auto h = [&](double u) {
        u = std::fabs(u) * kTableRes;
        const size_t i = static_cast<size_t>(u);
        if (i + 1 >= table.size()) return 0.0;
        const double f = u - static_cast<double>(i);
        return table[i] + (table[i + 1] - table[i]) * f;
    };

    const long n = static_cast<long>(input.size());
    const long outLen = static_cast<long>(std::ceil(static_cast<double>(n) * ratio));
    std::vector<float> out(static_cast<size_t>(outLen));
    const long reach = static_cast<long>(std::ceil(span));
    for (long o = 0; o < outLen; ++o) {
        const double t = static_cast<double>(o) / ratio;
        const long c = static_cast<long>(std::floor(t));
        double acc = 0;
        for (long j = std::max(0L, c - reach + 1); j <= std::min(n - 1, c + reach); ++j)
            acc += input[static_cast<size_t>(j)] * h(t - static_cast<double>(j));
        out[static_cast<size_t>(o)] = static_cast<float>(acc);
    }
    return out;
}

}  // namespace rvc
