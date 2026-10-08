#include "voice.h"

#include <cstdio>
#include <stdexcept>

namespace rvc {

using namespace detail;

VoicePtr loadVoice(const std::string& path, Precision precision) {
    auto v = std::make_shared<Voice>();
    v->dtype = precision == Precision::Float16 ? mx::float16 : mx::float32;
    v->weights = loadWeights(path, "voice", v->dtype);
    if (!v->weights.meta.count("config")) throw std::runtime_error("'" + path + "' has no voice config");
    v->config = SynthConfig::fromJson(Json::parse(v->weights.meta.at("config")));

    auto it = v->weights.arrays.find("index.vectors");
    if (it != v->weights.arrays.end()) {
        v->banks.push_back(mx::astype(it->second, mx::float32));
        v->bankWeights.push_back(1.0f);
        v->weights.arrays.erase(it);
    }
    VoiceInfo& info = v->info;
    info.name = v->weights.metaOr("name", "voice");
    info.version = v->config.version;
    info.info = v->weights.metaOr("info", "");
    info.mergedFrom = v->weights.metaOr("merged_from", "");
    info.sampleRate = v->config.sr;
    info.hasPitch = v->config.f0;
    info.numSpeakers = v->config.spkEmbedDim;
    info.indexSize = v->banks.empty() ? 0 : v->banks[0].shape(0);
    return v;
}

const VoiceInfo& voiceInfo(const Voice& voice) { return voice.info; }

std::string blendIncompatibility(const Voice& a, const Voice& b) {
    if (a.config.version != b.config.version) return "different RVC versions (" + a.config.version + " vs " + b.config.version + ")";
    if (a.config.sr != b.config.sr)
        return "different sample rates (" + std::to_string(a.config.sr) + " vs " + std::to_string(b.config.sr) + ")";
    if (a.config.f0 != b.config.f0) return "one voice uses pitch guidance and the other doesn't";
    if (a.config.architectureKey() != b.config.architectureKey()) return "different model sizes";
    return {};
}

VoicePtr blendVoices(const std::vector<VoicePtr>& voices, const std::vector<float>& weights) {
    if (voices.empty() || voices.size() != weights.size()) throw std::invalid_argument("need one weight per voice");
    double total = 0;
    for (float w : weights) total += w;
    if (total <= 0) throw std::invalid_argument("blend weights must sum to a positive number");
    for (size_t i = 1; i < voices.size(); ++i) {
        std::string why = blendIncompatibility(*voices[0], *voices[i]);
        if (!why.empty()) throw std::invalid_argument("Can't blend " + voices[0]->info.name + " and " + voices[i]->info.name + ": " + why);
    }
    std::vector<float> ws;
    for (float w : weights) ws.push_back(static_cast<float>(w / total));

    auto out = std::make_shared<Voice>();
    const Voice& ref = *voices[0];
    out->config = ref.config;
    out->dtype = ref.dtype;
    out->weights.path = "<blend>";
    std::vector<mx::array> toEval;
    for (const auto& [key, first] : ref.weights.arrays) {
        int rows = -1;
        if (key == "emb_g.weight") {
            rows = first.shape(0);
            for (const auto& v : voices) rows = std::min(rows, v->weights.at(key).shape(0));
        }
        std::optional<mx::array> acc;
        for (size_t i = 0; i < voices.size(); ++i) {
            if (ws[i] == 0.0f) continue;
            mx::array a = mx::astype(voices[i]->weights.at(key), mx::float32);
            if (rows >= 0) a = sliceAxis(a, 0, 0, rows);
            a = a * ws[i];
            acc = acc ? *acc + a : a;
        }
        mx::array value = mx::astype(*acc, first.dtype());
        out->weights.arrays.emplace(key, value);
        toEval.push_back(value);
    }
    mx::eval(toEval);
    if (out->weights.arrays.count("emb_g.weight")) out->config.spkEmbedDim = out->weights.at("emb_g.weight").shape(0);

    std::string recipe, name;
    int indexSize = 0;
    for (size_t i = 0; i < voices.size(); ++i) {
        const Voice& v = *voices[i];
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.3f", ws[i]);
        recipe += (i ? ", " : "") + v.info.name + ":" + buf;
        name += (i ? " + " : "") + v.info.name;
        if (ws[i] == 0.0f) continue;
        for (size_t b = 0; b < v.banks.size(); ++b) {
            out->banks.push_back(v.banks[b]);
            out->bankWeights.push_back(ws[i] * v.bankWeights[b]);
            indexSize += v.banks[b].shape(0);
        }
    }
    out->info = ref.info;
    out->info.name = name;
    out->info.mergedFrom = recipe;
    out->info.numSpeakers = out->config.spkEmbedDim;
    out->info.indexSize = indexSize;
    return out;
}

namespace detail {

namespace {
mx::array retrieve(const mx::array& q, const mx::array& bank, int k) {  // q (T, D), bank (N, D)
    k = std::min(k, bank.shape(0));
    mx::array bankSq = mx::sum(bank * bank, -1, false);
    std::vector<mx::array> outs;
    constexpr int kChunk = 512;
    for (int s = 0; s < q.shape(0); s += kChunk) {
        mx::array qc = sliceAxis(q, 0, s, std::min(q.shape(0), s + kChunk));
        mx::array d = mx::sum(qc * qc, -1, true) - 2.0f * mx::matmul(qc, mx::transpose(bank)) + mx::expand_dims(bankSq, 0);
        mx::array idx = sliceAxis(mx::argpartition(d, k - 1, -1), 1, 0, k);              // (t, k)
        mx::array dist = mx::maximum(mx::take_along_axis(d, idx, -1), mx::array(0.0f));
        mx::array w = 1.0f / mx::square(mx::maximum(dist, mx::array(1e-12f)));
        w = w / mx::sum(w, -1, true);
        mx::array rows = mx::reshape(mx::take(bank, mx::flatten(idx), 0), {idx.shape(0), k, bank.shape(1)});
        outs.push_back(mx::sum(rows * mx::expand_dims(w, -1), 1, false));
    }
    return mx::concatenate(outs, 0);
}
}  // namespace

mx::array retrieveBlended(const mx::array& feats, const std::vector<mx::array>& banks,
                          const std::vector<float>& weights, int k) {
    mx::array q = mx::astype(mx::squeeze(feats, 0), mx::float32);
    std::optional<mx::array> acc;
    double total = 0;
    for (float w : weights) total += w;
    for (size_t i = 0; i < banks.size(); ++i) {
        if (weights[i] == 0.0f) continue;
        mx::array r = retrieve(q, banks[i], k) * static_cast<float>(weights[i] / total);
        acc = acc ? *acc + r : r;
    }
    return mx::expand_dims(mx::astype(*acc, feats.dtype()), 0);
}

}  // namespace detail
}  // namespace rvc
