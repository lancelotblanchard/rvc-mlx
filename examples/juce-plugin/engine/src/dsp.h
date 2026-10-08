// Signal helpers mirroring rvc_mlx/dsp.py (scipy / numpy / librosa semantics), plus resampling.
#pragma once

#include <vector>

namespace rvc::detail {

/// scipy.signal.filtfilt with RVC's 5th-order 48 Hz Butterworth high-pass (odd extension, lfilter_zi init).
std::vector<double> highpass(const std::vector<double>& x);

/// numpy.pad(x, (left, right), mode="reflect"), including pads longer than the signal.
std::vector<double> reflectPad(const std::vector<double>& x, int left, int right);

/// librosa.feature.rms(y, frame_length, hop_length) (centered, zero padded).
std::vector<double> frameRms(const std::vector<double>& y, int frameLength, int hopLength);

/// torch.nn.functional.interpolate(mode="linear", align_corners=False) of a 1-D signal.
std::vector<double> interpLinear(const std::vector<double>& x, size_t size);

/// RVC's loudness-envelope mix (in place on `target`).
void changeRms(const std::vector<double>& source, int sourceRate, std::vector<float>& target, int targetRate,
               double rate);

/// f0 (Hz) -> RVC coarse pitch bins 1..255.
std::vector<int> coarseF0(const std::vector<double>& f0);

/// Chunk boundaries for long inputs (rvc_mlx.pipeline.Pipeline.split_points).
std::vector<long> splitPoints(const std::vector<double>& audio, int window, long tMax, long tCenter, long tQuery);

}  // namespace rvc::detail
