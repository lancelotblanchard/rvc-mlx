// HuBERT content encoder, RMVPE pitch estimator and RVC synthesizer: functional ports of rvc_mlx/{hubert,rmvpe,
// synthesizer}.py over converted weights. Activations are channels-last (B, T, C) as in the Python package.
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "json.h"
#include "weights.h"

namespace rvc::detail {

// ------------------------------------------------------------------------------------------------------------ HuBERT

class Hubert {
public:
    explicit Hubert(Weights weights);
    /// RVC content features for 16 kHz audio (1, L): v1 = layer 9 + final_proj (256-d), v2 = layer 12 (768-d).
    mx::array features(const mx::array& audio, const std::string& version) const;
    mx::array extract(const mx::array& audio, int outputLayer) const;
    mx::Dtype dtype() const { return dtype_; }

private:
    Weights w_;
    std::vector<std::array<int, 3>> convLayers_;
    int numHeads_ = 12, numLayers_ = 12, convPos_ = 128, convPosGroups_ = 16;
    bool hasFinalProj_ = true;
    mx::Dtype dtype_ = mx::float32;
};

// ------------------------------------------------------------------------------------------------------------- RMVPE

class Rmvpe {
public:
    explicit Rmvpe(Weights weights);
    /// f0 in Hz (0 = unvoiced), one value per 160 samples (1 + L / 160 values).
    std::vector<double> f0(const std::vector<float>& audio16k, float threshold = 0.03f) const;
    mx::array mel(const mx::array& audio) const;      // (1, 128, frames) log-mel
    mx::array salience(const mx::array& mel) const;   // (1, frames, 360)

private:
    mx::array convBlockRes(const std::string& p, const mx::array& x, int inC, int outC) const;
    mx::array biGru(const mx::array& x) const;

    Weights w_;
    int nBlocks_ = 4, enDeLayers_ = 5, interLayers_ = 4, enOutChannels_ = 16, nGru_ = 1;
    mx::array window_ = mx::array(0.0f);
};

// ------------------------------------------------------------------------------------------------------- Synthesizer

struct SynthConfig {
    int interChannels = 192, hiddenChannels = 192, filterChannels = 768, nHeads = 2, nLayers = 6, kernelSize = 3;
    std::string resblock = "1";
    std::vector<int> resblockKernelSizes, upsampleRates, upsampleKernelSizes;
    std::vector<std::vector<int>> resblockDilationSizes;
    int upsampleInitialChannel = 512, spkEmbedDim = 109, ginChannels = 256, sr = 40000;
    std::string version = "v2";
    bool f0 = true;

    static SynthConfig fromJson(const Json& j);
    int featureDim() const { return version == "v1" ? 256 : 768; }
    int hopLength() const;
    std::string architectureKey() const;  // everything that must match for blending
};

struct SynthInputs {
    mx::array phone = mx::array(0.0f);  // (1, T, featureDim)
    int length = 0;                     // valid frames
    std::optional<mx::array> pitch;     // (1, T) int32 coarse bins
    std::optional<mx::array> pitchf;    // (1, T) float32 Hz
    mx::array speaker = mx::array(0.0f);  // (1, ginChannels) speaker embedding
    float noiseScale = 0.66666f;        // prior temperature
    float nsfNoiseScale = 1.0f;
    std::optional<mx::array> priorNoise;  // (1, T, interChannels)
    std::optional<mx::array> nsfNoise;    // (1, T * hop, 1)
};

/// RVC `infer`: returns (1, T * hop) audio.
mx::array synthesize(const Weights& w, const SynthConfig& cfg, const SynthInputs& in);

/// NSF sine excitation (SineGen + noise), (1, T) Hz -> (1, T * upp, 1).
mx::array sineExcitation(const mx::array& f0, int upp, int sr, const std::optional<mx::array>& noise,
                         float noiseScale);

}  // namespace rvc::detail
