#pragma once

#include <JuceHeader.h>

#include <deque>
#include <functional>
#include <map>
#include <memory>

#include "RenderSettings.h"

class StreamingConverter;

/** A converted voice found in the models folder (read from the file header only). */
struct VoiceEntry
{
    juce::File file;
    juce::String name, version, mergedFrom;
    int sampleRate = 0;
    bool hasPitch = true;
    int indexSize = 0;
};

/**
    Process-wide owner of the MLX engine (one per process, shared by every plugin instance through
    juce::SharedResourcePointer). All MLX work runs on one worker thread, which also services the live
    (non-ARA) streaming converters before any queued offline render.
*/
class EngineHost : public juce::ChangeBroadcaster
{
public:
    EngineHost();
    ~EngineHost() override;

    enum class State { noFolder, loading, ready, error };

    //=== models folder ======================================================================================
    void setModelsFolder (const juce::File& folder);
    juce::File getModelsFolder() const;
    State getState() const;
    juce::String getStatusText() const;

    /** Voices in <models>/voices, sorted by name. Rescanned on setModelsFolder() / rescanVoices(). */
    juce::Array<VoiceEntry> getVoices() const;
    void rescanVoices();
    static bool readVoiceHeader (const juce::File& file, VoiceEntry& out);

    /** Explains why A and B can't be blended, or returns an empty string. Reads headers only. */
    juce::String blendProblem (const juce::String& voiceA, const juce::String& voiceB) const;

    //=== work (called from any non-audio thread) ============================================================
    using Job = std::function<void (rvc::Engine&)>;
    void submit (Job job);

    /** Worker thread only: the (cached) voice for a settings object, blending A/B as needed. */
    rvc::VoicePtr voiceFor (const RenderSettings& settings);

    //=== live streaming clients =============================================================================
    void addStreamClient (StreamingConverter*);
    void removeStreamClient (StreamingConverter*);
    void wakeUp() noexcept { wakeEvent.signal(); }

private:
    class Worker;
    rvc::VoicePtr loadCached (const juce::String& path);

    std::unique_ptr<Worker> worker;
    juce::WaitableEvent wakeEvent;

    mutable juce::CriticalSection lock;  // guards the fields below
    juce::File modelsFolder;
    State state = State::noFolder;
    juce::String statusText;
    juce::Array<VoiceEntry> voices;
    std::deque<Job> jobs;

    juce::CriticalSection streamLock;  // held while live clients are pumped (separate so the UI never waits on MLX)
    juce::Array<StreamingConverter*> streamClients;
    std::unique_ptr<juce::PropertiesFile> properties;

    // Worker-thread-only caches.
    std::unique_ptr<rvc::Engine> engine;
    juce::File engineFolder;
    std::map<juce::String, rvc::VoicePtr> voiceCache;
    std::map<juce::String, rvc::VoicePtr> blendCache;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (EngineHost)
};
