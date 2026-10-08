// Python-parity tests for the C++ engine.
//
//   python examples/juce-plugin/engine/tests/make_golden.py /tmp/rvc-golden [--full models/]
//   cmake -B build -DRVC_ENGINE_TESTS=ON examples/juce-plugin/engine && cmake --build build
//   ./build/rvc_parity_test /tmp/rvc-golden
//
// Each check runs one stage of the engine on the inputs the Python implementation used and compares against its
// output (max |error| relative to max |reference|).

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>

#include "dsp.h"
#include "models.h"
#include "rvc/rvc.h"
#include "voice.h"

using namespace rvc;
using namespace rvc::detail;

namespace {
int failures = 0, checks = 0;

std::vector<float> toVector(const mx::array& a) {
    mx::array f = mx::astype(mx::flatten(a), mx::float32);
    mx::eval(f);
    return std::vector<float>(f.data<float>(), f.data<float>() + f.size());
}

void compare(const std::string& name, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
    ++checks;
    if (got.size() != ref.size()) {
        std::printf("FAIL  %-34s size %zu != %zu\n", name.c_str(), got.size(), ref.size());
        ++failures;
        return;
    }
    double maxErr = 0, maxRef = 1e-9;
    for (size_t i = 0; i < ref.size(); ++i) {
        maxErr = std::max(maxErr, std::fabs(double(got[i]) - ref[i]));
        maxRef = std::max(maxRef, std::fabs(double(ref[i])));
    }
    const double rel = maxErr / maxRef;
    const bool ok = rel <= tol && std::isfinite(rel);
    if (!ok) ++failures;
    std::printf("%s  %-34s rel.err %.2e  (max|ref| %.3g, n=%zu)\n", ok ? "ok  " : "FAIL", name.c_str(), rel, maxRef, ref.size());
}

void compare(const std::string& name, const mx::array& got, const mx::array& ref, double tol) {
    if (got.shape() != ref.shape()) {
        ++checks;
        ++failures;
        std::printf("FAIL  %-34s shape %s != %s\n", name.c_str(), format(got.shape()).c_str(), format(ref.shape()).c_str());
        return;
    }
    compare(name, toVector(got), toVector(ref), tol);
}

void check(const std::string& name, bool ok, const std::string& detail = "") {
    ++checks;
    if (!ok) ++failures;
    std::printf("%s  %-34s %s\n", ok ? "ok  " : "FAIL", name.c_str(), detail.c_str());
}

void guarded(const std::string& name, const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        ++checks;
        ++failures;
        std::printf("FAIL  %-34s threw: %s\n", name.c_str(), e.what());
    }
}

double toneLevelDb(const std::vector<float>& x, double freq, double sr) {
    double re = 0, im = 0;
    const size_t a = x.size() / 4, b = 3 * x.size() / 4;  // avoid edges
    for (size_t i = a; i < b; ++i) {
        re += x[i] * std::cos(2 * M_PI * freq * i / sr);
        im += x[i] * std::sin(2 * M_PI * freq * i / sr);
    }
    return 20 * std::log10(2 * std::sqrt(re * re + im * im) / double(b - a) + 1e-12);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <golden dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    auto G = mx::load_safetensors(dir + "/golden.safetensors").first;
    auto g = [&](const std::string& k) { return G.at(k); };
    std::printf("MLX device: %s\n\n", mx::default_device() == mx::Device::gpu ? "gpu" : "cpu");

    guarded("hubert", [&] {
        Hubert hubert(loadWeights(dir + "/hubert.safetensors", "hubert", mx::float32));
        mx::array audio = mx::expand_dims(mx::slice(g("audio"), {0}, {32000}), 0);
        compare("hubert v2 features", hubert.features(audio, "v2"), g("hubert_v2"), 1e-3);
        compare("hubert v1 features", hubert.features(audio, "v1"), g("hubert_v1"), 1e-3);
    });

    guarded("rmvpe", [&] {
        Rmvpe rmvpe(loadWeights(dir + "/rmvpe.safetensors", "rmvpe", mx::float32));
        std::vector<float> audio = toVector(mx::slice(g("audio"), {0}, {32000}));
        mx::array a(audio.data(), {1, 32000}, mx::float32);
        mx::array mel = rmvpe.mel(a);
        compare("rmvpe log-mel", mel, g("rmvpe_mel"), 1e-4);
        compare("rmvpe salience", rmvpe.salience(g("rmvpe_mel")), g("rmvpe_salience"), 1e-3);
        std::vector<double> f0 = rmvpe.f0(audio);
        compare("rmvpe f0", std::vector<float>(f0.begin(), f0.end()), toVector(g("rmvpe_f0")), 1e-3);
    });

    guarded("sine", [&] {
        compare("nsf sine excitation", sineExcitation(g("sine_f0"), 160, 16000, g("sine_noise"), 1.0f), g("sine_out"), 2e-3);
    });

    for (const char* name : {"v2_f0", "v1_f0", "v2_nof0"}) {
        const std::string n = name;
        guarded("synth " + n, [&] {
            VoicePtr v = loadVoice(dir + "/" + n + ".safetensors");
            SynthInputs in;
            in.phone = g("synth_" + n + "_phone");
            in.length = in.phone.shape(1);
            in.pitch = g("synth_" + n + "_pitch");
            in.pitchf = g("synth_" + n + "_pitchf");
            in.priorNoise = g("synth_" + n + "_prior");
            in.nsfNoise = g("synth_" + n + "_nsf");
            in.speaker = sliceAxis(v->weights.at("emb_g.weight"), 0, 2, 3);
            compare("synthesizer " + n, synthesize(v->weights, v->config, in), g("synth_" + n + "_out"), 2e-3);
        });
    }

    guarded("pipeline", [&] {
        Engine engine(dir + "/hubert.safetensors", dir + "/rmvpe.safetensors");
        engine.chunking = {1, 1, 3, 4};
        std::vector<float> audio = toVector(g("audio"));
        for (const char* name : {"v2_f0", "v1_f0", "v2_nof0"}) {
            VoicePtr v = loadVoice(dir + "/" + std::string(name) + ".safetensors");
            ConvertOptions a{3.0f, 0.75f, 0.33f, 0.25f, 1, true};
            ConvertOptions b{-5.0f, 0.0f, 0.5f, 1.0f, 1, true};
            float last = 0;
            auto out = engine.convert(audio, *v, a, [&](float p) { last = p; return true; });
            compare(std::string("pipeline ") + name + " (index, protect)", out, toVector(g(std::string("pipe_") + name + "_a")), 3e-3);
            compare(std::string("pipeline ") + name + " (plain)", engine.convert(audio, *v, b),
                    toVector(g(std::string("pipe_") + name + "_b")), 3e-3);
            check(std::string("progress reaches 1 (") + name + ")", last == 1.0f);
        }
        VoicePtr v = loadVoice(dir + "/v2_f0.safetensors");
        ConvertOptions edge;
        edge.indexRate = 0.5f;
        edge.deterministic = true;
        compare("pipeline split at the last frame", engine.convert(toVector(g("edge_audio")), *v, edge),
                toVector(g("pipe_edge")), 3e-3);
        bool cancelled = false;
        try {
            engine.convert(audio, *v, {}, [](float) { return false; });
        } catch (const Cancelled&) {
            cancelled = true;
        }
        check("cancellation", cancelled);
        auto w = engine.convertWindow(std::vector<float>(audio.begin(), audio.begin() + 24000), *v, {}, 4000);
        check("convertWindow output size", w.size() == 4000);
    });

    guarded("blend", [&] {
        VoicePtr a = loadVoice(dir + "/v2_f0.safetensors"), b = loadVoice(dir + "/v2_f0_b.safetensors");
        VoicePtr mix = blendVoices({a, b}, {3.0f, 7.0f});
        for (const char* key : {"dec.ups.0.weight", "enc_p.emb_phone.weight", "emb_g.weight", "flow.flows.2.enc.cond_layer.weight"})
            compare(std::string("blend ") + key, mix->weights.at(key), g(std::string("blend_") + key), 1e-6);
        check("blend metadata", voiceInfo(*mix).mergedFrom == "v2_f0:0.300, v2_f0_b:0.700", voiceInfo(*mix).mergedFrom);
        compare("blended retrieval", retrieveBlended(g("retrieve_q"), {a->banks[0], b->banks[0]}, {0.3f, 0.7f}),
                g("retrieve_out"), 1e-4);
        VoicePtr v1 = loadVoice(dir + "/v1_f0.safetensors");
        check("blend rejects v1 + v2", !blendIncompatibility(*a, *v1).empty(), blendIncompatibility(*a, *v1));
    });

    guarded("dsp", [&] {
        std::vector<float> x = toVector(g("dsp_x"));
        std::vector<double> hp = highpass(std::vector<double>(x.begin(), x.end()));
        compare("highpass (filtfilt)", std::vector<float>(hp.begin(), hp.end()), toVector(g("dsp_highpass")), 1e-5);
        std::vector<float> t = toVector(g("dsp_rms_target"));
        changeRms(std::vector<double>(x.begin(), x.end()), 16000, t, 40000, 0.25);
        compare("change_rms", t, toVector(g("dsp_rms_out")), 1e-4);
        auto r = reflectPad({1, 2, 3}, 5, 4);
        check("reflect pad (long)", r == std::vector<double>{2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3});  // == np.pad(..., "reflect")
    });

    guarded("resample", [&] {
        const double sr = 44100;
        std::vector<float> tone(44100), high(44100);
        for (size_t i = 0; i < tone.size(); ++i) {
            tone[i] = 0.5f * std::sin(2 * M_PI * 440 * i / sr);
            high[i] = 0.5f * std::sin(2 * M_PI * 10000 * i / sr);
        }
        auto down = resample(tone, sr, 16000);
        auto back = resample(down, 16000, sr);
        check("resample length", down.size() == 16000 && back.size() == 44100);
        double err = 0;
        for (size_t i = 2000; i < 42000; ++i) err = std::max(err, double(std::fabs(back[i] - tone[i])));
        check("resample round trip (440 Hz)", err < 2e-3, "max err " + std::to_string(err));
        const double kept = toneLevelDb(down, 440, 16000), alias = toneLevelDb(resample(high, sr, 16000), 6000, 16000);
        check("resample passband level", std::fabs(kept - 20 * std::log10(0.5)) < 0.05, std::to_string(kept) + " dB");
        check("resample anti-aliasing", alias < -70, "10 kHz alias at " + std::to_string(alias) + " dB");
    });

    if (std::filesystem::exists(dir + "/golden_full.safetensors")) {
        guarded("full-size", [&] {
            auto F = mx::load_safetensors(dir + "/golden_full.safetensors");
            const std::string models = std::filesystem::path(dir) / "full";
            auto f = [&](const std::string& k) { return F.first.at(k); };
            std::vector<float> audio = toVector(f("audio"));
            auto engine = Engine::fromFolder(models);
            Hubert hubert(loadWeights(models + "/hubert.safetensors", "hubert", mx::float32));
            mx::array a(audio.data(), {1, static_cast<int>(audio.size())}, mx::float32);
            compare("full hubert v2", hubert.features(a, "v2"), f("hubert_v2"), 2e-3);
            compare("full rmvpe f0", engine->pitch(audio), toVector(f("rmvpe_f0")), 2e-3);
            std::string voicePath;
            for (auto& e : std::filesystem::directory_iterator(models + "/voices"))
                if (voicePath.empty() || e.path().string() < voicePath) voicePath = e.path().string();
            ConvertOptions o;
            o.pitch = 2;
            o.deterministic = true;
            compare("full pipeline (40k voice)", engine->convert(audio, *loadVoice(voicePath), o), toVector(f("pipe")), 5e-3);
        });
    }

    std::printf("\n%d / %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
