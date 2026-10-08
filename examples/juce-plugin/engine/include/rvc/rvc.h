// rvc-mlx C++ engine: RVC voice conversion on MLX (Metal on Apple silicon).
//
// A dependency-light port of the Python package (`rvc_mlx`) used by the JUCE plugin. It reads the same converted
// `.safetensors` files and reproduces the Python pipeline numerically (see tests/parity_test.cpp).
//
//     auto engine = rvc::Engine::fromFolder("models/");
//     auto voice  = rvc::loadVoice("models/voices/alice.safetensors");
//     auto out    = engine->convert(rvc::resample(input, 44100, 16000), *voice, {.pitch = 12});
//     // `out` is at rvc::voiceInfo(*voice).sampleRate
//
// Thread-safety: an Engine and the Voices are immutable after construction; convert() may be called from several
// threads, although on one GPU it is best to serialise calls.
#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rvc {

enum class Precision { Float32, Float16 };

struct VoiceInfo {
    std::string name;
    std::string version;     // "v1" or "v2"
    std::string info;        // free text from training (epochs, ...)
    std::string mergedFrom;  // "alice:0.700, bob:0.300" for blended voices
    int sampleRate = 0;      // output sample rate (32000 / 40000 / 48000)
    bool hasPitch = true;    // false for "nono" voices (no pitch guidance)
    int numSpeakers = 1;
    int indexSize = 0;       // retrieval vectors (summed over banks for blends)
};

class Voice;
using VoicePtr = std::shared_ptr<const Voice>;

/// Load a converted voice. Throws std::runtime_error with a readable message on failure.
VoicePtr loadVoice(const std::string& path, Precision precision = Precision::Float32);

const VoiceInfo& voiceInfo(const Voice& voice);

/// Returns an empty string when the voices can be blended, otherwise the reason they can't.
std::string blendIncompatibility(const Voice& a, const Voice& b);

/// Weighted interpolation of synthesizer weights (RVC "ckpt merge"). Each voice's retrieval bank is kept and searched
/// separately, with results mixed by the same weights. Weights are normalised to sum to 1.
VoicePtr blendVoices(const std::vector<VoicePtr>& voices, const std::vector<float>& weights);

struct ConvertOptions {
    float pitch = 0.0f;         // transposition, semitones
    float indexRate = 0.75f;    // 0..1, pull towards the voice's training features
    float protect = 0.33f;      // 0..0.5, protect unvoiced consonants (0.5 = off)
    float rmsMixRate = 0.25f;   // 0..1, 1 = converted loudness, 0 = input loudness
    int speaker = 0;
    bool deterministic = false; // disable sampling noise (tests, A/B)
};

/// Chunking geometry in seconds (see rvc_mlx.pipeline.PipelineConfig).
struct ChunkingConfig {
    int xPad = 1, xQuery = 6, xCenter = 38, xMax = 41;
};

/// Progress callback: receives 0..1, returns false to cancel (convert() then throws rvc::Cancelled).
using Progress = std::function<bool(float)>;

struct Cancelled : std::exception {
    const char* what() const noexcept override { return "conversion cancelled"; }
};

class Engine {
public:
    Engine(const std::string& hubertPath, const std::string& rmvpePath, Precision precision = Precision::Float32);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// Loads `hubert.safetensors` and `rmvpe.safetensors` from a models folder.
    static std::unique_ptr<Engine> fromFolder(const std::string& modelsDir, Precision precision = Precision::Float32);

    /// The full RVC pipeline over 16 kHz mono audio. Returns audio at the voice's sample rate, peak-limited to 0.99.
    std::vector<float> convert(const std::vector<float>& audio16k, const Voice& voice,
                               const ConvertOptions& options = {}, const Progress& progress = {}) const;

    /// Streaming helper: converts a window of real audio context (no reflect padding, chunking or loudness mixing)
    /// and returns its last `outputSamples` samples at the voice's sample rate.
    std::vector<float> convertWindow(const std::vector<float>& window16k, const Voice& voice,
                                     const ConvertOptions& options, int outputSamples) const;

    /// RMVPE pitch track: one f0 value (Hz, 0 = unvoiced) per 10 ms frame.
    std::vector<float> pitch(const std::vector<float>& audio16k) const;

    ChunkingConfig chunking;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

/// Band-limited (Kaiser-windowed sinc) sample-rate conversion.
std::vector<float> resample(const std::vector<float>& input, double fromRate, double toRate);

}  // namespace rvc
