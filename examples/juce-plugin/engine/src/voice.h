#pragma once

#include "models.h"
#include "rvc/rvc.h"

namespace rvc {

class Voice {
public:
    VoiceInfo info;
    detail::SynthConfig config;
    detail::Weights weights;                 // synthesizer parameters (MLX layout)
    std::vector<detail::mx::array> banks;    // retrieval feature banks, (N, featureDim) each
    std::vector<float> bankWeights;          // mixing weight per bank (sums to 1)
    detail::mx::Dtype dtype = detail::mx::float32;
};

namespace detail {
/// RVC's index retrieval against several banks: sum_i w_i * retrieve(feats, bank_i). feats: (1, T, D).
mx::array retrieveBlended(const mx::array& feats, const std::vector<mx::array>& banks,
                          const std::vector<float>& weights, int k = 8);
}  // namespace detail

}  // namespace rvc
