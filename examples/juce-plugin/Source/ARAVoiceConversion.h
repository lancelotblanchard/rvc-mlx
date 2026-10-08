#pragma once

#include <JuceHeader.h>

#if JucePlugin_Enable_ARA

#include <map>
#include <set>

#include "EngineHost.h"

/**
    One audio source converted with one set of settings, at the playback sample rate. Rendered once in the
    background, then read lock-free by the playback renderers.
*/
struct RenderedClip
{
    enum State { queued, reading, converting, ready, failed };

    std::atomic<int> state { queued };
    std::atomic<float> progress { 0.0f };
    std::atomic<bool> cancelled { false };

    juce::AudioBuffer<float> dry;  // original audio, valid once state >= converting
    std::vector<float> wet;        // converted mono audio, valid once state == ready
    juce::String error;            // set before state becomes failed
};

/**
    Per-document cache of rendered clips, keyed by (audio source, settings, playback rate). Renders run on the
    EngineHost thread; cached audio is dropped when the host changes a source's samples or destroys it.
*/
class RenderCache : private juce::ARAAudioSource::Listener
{
public:
    explicit RenderCache (EngineHost& host);
    ~RenderCache() override;

    /** Message thread: returns the clip for these settings, queueing a render if needed. */
    std::shared_ptr<const RenderedClip> request (juce::ARAAudioSource* source, const RenderSettings& settings, double playbackRate);

    /** Message thread: forget clips nobody references any more (cancelling their renders). */
    void collectGarbage();

private:
    using Key = std::pair<juce::ARAAudioSource*, juce::int64>;

    void doUpdateAudioSourceContent (juce::ARAAudioSource* source, juce::ARAContentUpdateScopes scopeFlags) override;
    void didEnableAudioSourceSamplesAccess (juce::ARAAudioSource* source, bool enable) override;
    void willDestroyAudioSource (juce::ARAAudioSource* source) override;
    void drop (juce::ARAAudioSource* source, bool onlyFailed = false);

    EngineHost& host;
    std::map<Key, std::shared_ptr<RenderedClip>> clips;
    std::set<juce::ARAAudioSource*> listening;
};

//==============================================================================================================
class RvcPlaybackRenderer : public juce::ARAPlaybackRenderer
{
public:
    using juce::ARAPlaybackRenderer::ARAPlaybackRenderer;

    void prepareToPlay (double sampleRate, int maximumSamplesPerBlock, int numChannels,
                        juce::AudioProcessor::ProcessingPrecision precision, AlwaysNonRealtime alwaysNonRealtime) override;
    bool processBlock (juce::AudioBuffer<float>& buffer, juce::AudioProcessor::Realtime realtime,
                       const juce::AudioPlayHead::PositionInfo& positionInfo) noexcept override;

    /** Message thread (called ~10 times a second by the processor): render what this instance's regions need. */
    void update (const RenderSettings& settings, float mix, float gain);

    struct Status
    {
        int clips = 0, ready = 0;
        float progress = 1.0f;
        juce::String error;
    };
    Status getStatus() const;

private:
    struct Table
    {
        std::vector<std::pair<const juce::ARAAudioSource*, std::shared_ptr<const RenderedClip>>> clips;
    };

    std::shared_ptr<const Table> table;  // swapped with std::atomic_store / read with std::atomic_load
    std::atomic<double> playbackRate { 0.0 };
    std::atomic<float> mix { 1.0f }, gain { 1.0f };
};

//==============================================================================================================
class RvcDocumentController : public juce::ARADocumentControllerSpecialisation
{
public:
    using juce::ARADocumentControllerSpecialisation::ARADocumentControllerSpecialisation;

    RenderCache& getCache() { return cache; }

protected:
    juce::ARAPlaybackRenderer* doCreatePlaybackRenderer() noexcept override;
    bool doRestoreObjectsFromStream (juce::ARAInputStream&, const juce::ARARestoreObjectsFilter*) noexcept override;
    bool doStoreObjectsToStream (juce::ARAOutputStream&, const juce::ARAStoreObjectsFilter*) noexcept override;

private:
    juce::SharedResourcePointer<EngineHost> host;
    RenderCache cache { *host };
};

#endif
