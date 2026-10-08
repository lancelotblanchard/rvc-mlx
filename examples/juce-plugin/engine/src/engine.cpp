#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "dsp.h"
#include "voice.h"

namespace rvc {

using namespace detail;

struct Engine::Impl {
    Hubert hubert;
    Rmvpe rmvpe;
};

namespace {
constexpr int kSr = 16000;
constexpr int kWindow = 160;  // samples per f0 frame

std::string findFile(const std::string& dir, std::initializer_list<const char*> names) {
    for (const char* n : names) {
        auto p = std::filesystem::path(dir) / n;
        if (std::filesystem::exists(p)) return p.string();
    }
    throw std::runtime_error("No " + std::string(*names.begin()) + " in '" + dir + "'. Convert it with `rvc-mlx convert-base`.");
}

struct PitchTrack {
    std::vector<int> coarse;
    std::vector<float> hz;
};

PitchTrack pitchTrack(const Rmvpe& rmvpe, const std::vector<double>& audio, float semitones) {
    std::vector<float> x(audio.begin(), audio.end());
    std::vector<double> f0 = rmvpe.f0(x);
    const double factor = std::pow(2.0, semitones / 12.0);
    for (double& v : f0) v *= factor;
    PitchTrack t;
    t.coarse = coarseF0(f0);
    t.hz.assign(f0.begin(), f0.end());
    return t;
}

/// One RVC `vc` call: features -> retrieval -> protect -> synthesis. Returns audio at the voice rate.
std::vector<float> vc(const Hubert& hubert, const Voice& voice, const ConvertOptions& opt, const double* audio,
                      size_t n, const int* pitch, const float* pitchf, size_t pitchLen) {
    const bool hasPitch = voice.config.f0;
    std::vector<float> a(audio, audio + n);
    mx::array x(a.data(), {1, static_cast<int>(n)}, mx::float32);
    mx::array feats = mx::astype(hubert.features(x, voice.config.version), mx::float32);
    mx::array feats0 = feats;
    if (!voice.banks.empty() && opt.indexRate != 0.0f)
        feats = retrieveBlended(feats, voice.banks, voice.bankWeights) * opt.indexRate + feats * (1.0f - opt.indexRate);
    feats = mx::repeat(feats, 2, 1);
    feats0 = mx::repeat(feats0, 2, 1);

    int pLen = static_cast<int>(n / kWindow);
    pLen = std::min(pLen, feats.shape(1));
    if (hasPitch) pLen = std::min(pLen, static_cast<int>(pitchLen));
    feats = sliceAxis(feats, 1, 0, pLen);

    SynthInputs in;
    if (hasPitch) {
        std::vector<int32_t> p(pitch, pitch + pLen);
        std::vector<float> pf(pitchf, pitchf + pLen);
        in.pitch = mx::array(p.data(), {1, pLen}, mx::int32);
        in.pitchf = mx::array(pf.data(), {1, pLen}, mx::float32);
        if (opt.protect < 0.5f) {
            std::vector<float> keep(static_cast<size_t>(pLen));
            for (int i = 0; i < pLen; ++i) keep[static_cast<size_t>(i)] = pitchf[i] < 1.0f ? opt.protect : 1.0f;
            mx::array k(keep.data(), {1, pLen, 1}, mx::float32);
            feats = feats * k + sliceAxis(feats0, 1, 0, pLen) * (1.0f - k);
        }
    }
    in.phone = mx::astype(feats, voice.dtype);
    in.length = pLen;
    const int spk = std::clamp(opt.speaker, 0, voice.config.spkEmbedDim - 1);
    in.speaker = sliceAxis(voice.weights.at("emb_g.weight"), 0, spk, spk + 1);
    if (opt.deterministic) {
        in.noiseScale = 0.0f;
        in.nsfNoiseScale = 0.0f;
    }
    mx::array out = mx::astype(synthesize(voice.weights, voice.config, in), mx::float32);
    mx::eval(out);
    const float* d = out.data<float>();
    return std::vector<float>(d, d + out.size());
}
}  // namespace

Engine::Engine(const std::string& hubertPath, const std::string& rmvpePath, Precision precision)
    : impl_(new Impl{Hubert(loadWeights(hubertPath, "hubert", precision == Precision::Float16 ? mx::float16 : mx::float32)),
                     Rmvpe(loadWeights(rmvpePath, "rmvpe", mx::float32))}) {}

Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::fromFolder(const std::string& dir, Precision precision) {
    return std::make_unique<Engine>(findFile(dir, {"hubert.safetensors", "hubert_base.safetensors", "contentvec.safetensors"}),
                                    findFile(dir, {"rmvpe.safetensors"}), precision);
}

std::vector<float> Engine::pitch(const std::vector<float>& audio16k) const {
    std::vector<double> f0 = impl_->rmvpe.f0(audio16k);
    return std::vector<float>(f0.begin(), f0.end());
}

std::vector<float> Engine::convert(const std::vector<float>& audio16k, const Voice& voice,
                                   const ConvertOptions& opt, const Progress& progress) const {
    if (audio16k.size() < kSr / 10) throw std::invalid_argument("audio must be at least 0.1 s long");
    const int tgtSr = voice.config.sr;
    const long tPad = static_cast<long>(kSr) * chunking.xPad, tPadTgt = static_cast<long>(tgtSr) * chunking.xPad;
    const long tPad2 = 2 * tPad;

    std::vector<double> audio = highpass(std::vector<double>(audio16k.begin(), audio16k.end()));
    std::vector<long> opt_ts = splitPoints(audio, kWindow, static_cast<long>(kSr) * chunking.xMax,
                                           static_cast<long>(kSr) * chunking.xCenter, static_cast<long>(kSr) * chunking.xQuery);
    std::vector<double> pad = reflectPad(audio, static_cast<int>(tPad), static_cast<int>(tPad));
    const size_t pLen = pad.size() / kWindow;

    PitchTrack track;
    if (voice.config.f0) {
        track = pitchTrack(impl_->rmvpe, pad, opt.pitch);
        track.coarse.resize(std::min(track.coarse.size(), pLen));
        track.hz.resize(std::min(track.hz.size(), pLen));
    }

    std::vector<float> out;
    const size_t nChunks = opt_ts.size() + 1;
    auto runChunk = [&](long s, long e, long pitchStart) {
        const int* p = nullptr;
        const float* pf = nullptr;
        size_t plen = 0;
        if (voice.config.f0) {
            const size_t ps = std::min(static_cast<size_t>(pitchStart), track.coarse.size());
            p = track.coarse.data() + ps;
            pf = track.hz.data() + ps;
            plen = track.coarse.size() - ps;
        }
        std::vector<float> chunk = vc(impl_->hubert, voice, opt, pad.data() + s, static_cast<size_t>(e - s), p, pf, plen);
        if (static_cast<long>(chunk.size()) > 2 * tPadTgt)
            out.insert(out.end(), chunk.begin() + tPadTgt, chunk.end() - tPadTgt);
        if (progress && !progress(static_cast<float>(out.size()) / std::max<double>(1.0, audio.size() * double(tgtSr) / kSr)))
            throw Cancelled();
    };

    long s = 0;
    long t = -1;
    for (size_t i = 0; i < opt_ts.size(); ++i) {
        t = opt_ts[i] / kWindow * kWindow;
        // The reference also bounds the pitch slice at (t + tPad2) / window; vc() truncates to the feature length.
        runChunk(s, std::min(static_cast<long>(pad.size()), t + tPad2 + kWindow), s / kWindow);  // slices clamp, like numpy
        s = t;
    }
    runChunk(t < 0 ? 0 : t, static_cast<long>(pad.size()), t < 0 ? 0 : t / kWindow);
    (void)nChunks;

    if (opt.rmsMixRate != 1.0f) changeRms(audio, kSr, out, tgtSr, opt.rmsMixRate);
    float peak = 0;
    for (float v : out) peak = std::max(peak, std::fabs(v));
    if (peak / 0.99f > 1.0f)
        for (float& v : out) v /= peak / 0.99f;
    if (progress) progress(1.0f);
    return out;
}

std::vector<float> Engine::convertWindow(const std::vector<float>& window16k, const Voice& voice,
                                         const ConvertOptions& opt, int outputSamples) const {
    std::vector<double> audio(window16k.begin(), window16k.end());
    PitchTrack track;
    if (voice.config.f0) track = pitchTrack(impl_->rmvpe, audio, opt.pitch);
    std::vector<float> y = vc(impl_->hubert, voice, opt, audio.data(), audio.size(), track.coarse.data(),
                              track.hz.data(), track.coarse.size());
    std::vector<float> out(static_cast<size_t>(outputSamples), 0.0f);
    const size_t take = std::min(y.size(), out.size());
    std::copy(y.end() - static_cast<long>(take), y.end(), out.end() - static_cast<long>(take));
    return out;
}

}  // namespace rvc
