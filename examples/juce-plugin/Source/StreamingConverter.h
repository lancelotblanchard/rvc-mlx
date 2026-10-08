#pragma once

#include <JuceHeader.h>

#include <atomic>

#include "RenderSettings.h"

class EngineHost;

/**
    Live voice conversion for hosts without ARA.

    The audio thread pushes mono input into a lock-free FIFO and pops converted audio from another. The engine
    thread converts overlapping windows (`kContextSeconds` of history + one hop) and stitches consecutive hops with
    SOLA: it searches a few milliseconds for the best-correlated alignment with the previous hop's tail, then
    crossfades. The output FIFO is pre-filled so the engine thread has one full hop of time per block; the total
    delay is reported to the host as plug-in latency.
*/
class StreamingConverter
{
public:
    static constexpr double kHopSeconds = 0.30;
    static constexpr double kContextSeconds = 1.20;
    static constexpr double kCrossfadeSeconds = 0.04;
    static constexpr double kSearchSeconds = 0.012;

    explicit StreamingConverter (EngineHost& host);
    ~StreamingConverter();

    /** Not on the audio thread. Resets all state. */
    void prepare (double sampleRate, int maximumBlockSize);
    void release();
    int getLatencySamples() const noexcept { return latency; }

    /** Audio thread: `numSamples` mono samples in, the same number of (delayed) converted samples out. */
    void process (const float* input, float* output, int numSamples) noexcept;

    /** Message thread. */
    void setSettings (const RenderSettings& settings);

    /** Fraction of real time the engine needs per hop (> 1 means it can't keep up). */
    float getLoad() const noexcept { return load.load(); }
    int getUnderruns() const noexcept { return underruns.load(); }

    /** Engine thread (called by EngineHost): converts every complete hop waiting in the input FIFO. */
    void pump (rvc::Engine& engine, EngineHost& host);

private:
    void convertHop (rvc::Engine& engine, EngineHost& host, float* out);

    EngineHost& host;
    std::atomic<bool> prepared { false };
    double sampleRate = 48000.0;
    int hop = 0, crossfade = 0, search = 0, context = 0, margin = 0, latency = 0;

    juce::AbstractFifo inFifo { 1 }, outFifo { 1 };
    std::vector<float> inBuffer, outBuffer;

    // Engine-thread state.
    std::vector<float> history, hopIn, hopOut, previousTail, fade;
    bool havePreviousTail = false;

    juce::SpinLock settingsLock;
    RenderSettings settings;
    std::atomic<float> load { 0.0f };
    std::atomic<int> underruns { 0 };
};
