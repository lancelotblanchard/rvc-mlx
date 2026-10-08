#pragma once

#include <JuceHeader.h>

#include "EngineHost.h"
#include "StreamingConverter.h"

namespace ParamIDs
{
inline constexpr auto pitch = "pitch";
inline constexpr auto blend = "blend";
inline constexpr auto index = "index";
inline constexpr auto protect = "protect";
inline constexpr auto mix = "mix";
inline constexpr auto output = "output";
} // namespace ParamIDs

class RvcProcessor : public juce::AudioProcessor,
                     #if JucePlugin_Enable_ARA
                     public juce::AudioProcessorARAExtension,
                     #endif
                     private juce::Timer
{
public:
    RvcProcessor();
    ~RvcProcessor() override;

    //=== AudioProcessor =====================================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override;
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //=== editor-facing ======================================================================================
    juce::AudioProcessorValueTreeState& getParameters() { return parameters; }
    EngineHost& getEngineHost() { return *host; }

    enum class Slot { A, B };
    juce::String getVoicePath (Slot slot) const;
    void setVoicePath (Slot slot, const juce::String& path);

    bool isUsingARA() const;

    /** One line describing what the plug-in is doing, for the editor's status pill. */
    struct Status
    {
        enum class Tone { idle, busy, ok, warning } tone = Tone::idle;
        juce::String text;
        float progress = -1.0f;  // 0..1 while rendering, -1 otherwise
    };
    Status getStatus() const;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    RenderSettings currentSettings() const;
    void timerCallback() override;

    #if JucePlugin_Enable_ARA
    void didBindToARA() noexcept override;
    #endif

    juce::SharedResourcePointer<EngineHost> host;
    juce::AudioProcessorValueTreeState parameters;
    StreamingConverter streaming { *host };

    // Live path buffers (allocated in prepareToPlay).
    std::vector<float> monoIn, wetOut;
    juce::AudioBuffer<float> dryDelay;
    int dryDelayPos = 0;
    juce::SmoothedValue<float> mixSmoothed, gainSmoothed;

    // Settings debouncing for ARA re-renders: only commit once the controls have been still for a moment.
    juce::int64 pendingHash = 0, committedHash = 0;
    double pendingSince = 0;
    RenderSettings committed;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RvcProcessor)
};
