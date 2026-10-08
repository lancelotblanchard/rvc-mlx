#include "PluginProcessor.h"

#include "ARAVoiceConversion.h"
#include "PluginEditor.h"

namespace
{
constexpr double kCommitDelayMs = 350.0;

float dbToGain (float db) { return juce::Decibels::decibelsToGain (db, -60.0f); }
} // namespace

RvcProcessor::RvcProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      parameters (*this, nullptr, "RVCMorph", createLayout())
{
    parameters.state.setProperty ("voiceA", "", nullptr);
    parameters.state.setProperty ("voiceB", "", nullptr);
    startTimerHz (10);
}

RvcProcessor::~RvcProcessor()
{
    stopTimer();
    streaming.release();
}

juce::AudioProcessorValueTreeState::ParameterLayout RvcProcessor::createLayout()
{
    using namespace juce;
    auto pct = [] (float v, int) { return String (roundToInt (v * 100.0f)) + " %"; };
    std::vector<std::unique_ptr<RangedAudioParameter>> p;
    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { ParamIDs::pitch, 1 }, "Pitch",
        NormalisableRange<float> (-24.0f, 24.0f, 1.0f), 0.0f,
        AudioParameterFloatAttributes().withLabel ("st").withStringFromValueFunction ([] (float v, int)
        { return (v > 0 ? "+" : "") + String (roundToInt (v)) + " st"; })));
    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { ParamIDs::blend, 1 }, "Blend",
        NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.0f, AudioParameterFloatAttributes().withStringFromValueFunction (pct)));
    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { ParamIDs::index, 1 }, "Index",
        NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.75f, AudioParameterFloatAttributes().withStringFromValueFunction (pct)));
    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { ParamIDs::protect, 1 }, "Protect",
        NormalisableRange<float> (0.0f, 0.5f, 0.01f), 0.33f, AudioParameterFloatAttributes().withStringFromValueFunction ([] (float v, int)
        { return String (v, 2); })));
    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { ParamIDs::mix, 1 }, "Mix",
        NormalisableRange<float> (0.0f, 1.0f, 0.01f), 1.0f, AudioParameterFloatAttributes().withStringFromValueFunction (pct)));
    p.push_back (std::make_unique<AudioParameterFloat> (ParameterID { ParamIDs::output, 1 }, "Output",
        NormalisableRange<float> (-24.0f, 12.0f, 0.1f), 0.0f, AudioParameterFloatAttributes().withLabel ("dB")
        .withStringFromValueFunction ([] (float v, int) { return String (v, 1) + " dB"; })));
    return { p.begin(), p.end() };
}

//==============================================================================================================
bool RvcProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
        return false;
    return layouts.getMainInputChannelSet() == out;
}

void RvcProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    #if JucePlugin_Enable_ARA
    if (prepareToPlayForARA (sampleRate, samplesPerBlock, getMainBusNumOutputChannels(), getProcessingPrecision()))
    {
        setLatencySamples (0);  // ARA renders ahead of time
        return;
    }
    #endif
    streaming.prepare (sampleRate, samplesPerBlock);
    monoIn.assign ((size_t) samplesPerBlock, 0.0f);
    wetOut.assign ((size_t) samplesPerBlock, 0.0f);
    dryDelay.setSize (getTotalNumInputChannels(), streaming.getLatencySamples() + 1);
    dryDelay.clear();
    dryDelayPos = 0;
    mixSmoothed.reset (sampleRate, 0.05);
    gainSmoothed.reset (sampleRate, 0.05);
    mixSmoothed.setCurrentAndTargetValue (parameters.getRawParameterValue (ParamIDs::mix)->load());
    gainSmoothed.setCurrentAndTargetValue (dbToGain (parameters.getRawParameterValue (ParamIDs::output)->load()));
    setLatencySamples (streaming.getLatencySamples());
}

void RvcProcessor::releaseResources()
{
    #if JucePlugin_Enable_ARA
    releaseResourcesForARA();
    #endif
    streaming.release();
}

double RvcProcessor::getTailLengthSeconds() const
{
    #if JucePlugin_Enable_ARA
    double tail = 0;
    if (getTailLengthSecondsForARA (tail))
        return tail;
    #endif
    return 0.0;
}

#if JucePlugin_Enable_ARA
void RvcProcessor::didBindToARA() noexcept
{
    AudioProcessorARAExtension::didBindToARA();
    streaming.release();
    setLatencySamples (0);
}
#endif

bool RvcProcessor::isUsingARA() const
{
    #if JucePlugin_Enable_ARA
    return isBoundToARA();
    #else
    return false;
    #endif
}

void RvcProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    #if JucePlugin_Enable_ARA
    if (processBlockForARA (buffer, isNonRealtime() ? Realtime::no : Realtime::yes, getPlayHead()))
        return;
    #endif

    // Live path: mono conversion, dry signal delayed by the same latency so the mix stays phase-aligned.
    const int n = buffer.getNumSamples(), channels = buffer.getNumChannels();
    if (n > (int) monoIn.size() || dryDelay.getNumSamples() < 2)
    {
        buffer.clear();
        return;
    }
    juce::FloatVectorOperations::clear (monoIn.data(), n);
    for (int c = 0; c < channels; ++c)
        juce::FloatVectorOperations::addWithMultiply (monoIn.data(), buffer.getReadPointer (c), 1.0f / (float) channels, n);
    streaming.process (monoIn.data(), wetOut.data(), n);

    mixSmoothed.setTargetValue (parameters.getRawParameterValue (ParamIDs::mix)->load());
    gainSmoothed.setTargetValue (dbToGain (parameters.getRawParameterValue (ParamIDs::output)->load()));
    const int delayLength = dryDelay.getNumSamples();
    for (int i = 0; i < n; ++i)
    {
        const float wetAmount = mixSmoothed.getNextValue(), g = gainSmoothed.getNextValue();
        const int readPos = (dryDelayPos + 1) % delayLength;  // oldest sample = exactly `latency` samples ago
        for (int c = 0; c < channels; ++c)
        {
            auto* io = buffer.getWritePointer (c);
            const int dc = juce::jmin (c, dryDelay.getNumChannels() - 1);
            dryDelay.setSample (dc, dryDelayPos, io[i]);
            const float dry = dryDelay.getSample (dc, readPos);
            io[i] = g * (wetAmount * wetOut[(size_t) i] + (1.0f - wetAmount) * dry);
        }
        dryDelayPos = readPos;
    }
}

//==============================================================================================================
juce::String RvcProcessor::getVoicePath (Slot slot) const
{
    return parameters.state.getProperty (slot == Slot::A ? "voiceA" : "voiceB").toString();
}

void RvcProcessor::setVoicePath (Slot slot, const juce::String& path)
{
    parameters.state.setProperty (slot == Slot::A ? "voiceA" : "voiceB", path, nullptr);
}

RenderSettings RvcProcessor::currentSettings() const
{
    RenderSettings s;
    s.voiceA = getVoicePath (Slot::A);
    s.voiceB = getVoicePath (Slot::B);
    s.blend = parameters.getRawParameterValue (ParamIDs::blend)->load();
    s.pitch = parameters.getRawParameterValue (ParamIDs::pitch)->load();
    s.indexRate = parameters.getRawParameterValue (ParamIDs::index)->load();
    s.protect = parameters.getRawParameterValue (ParamIDs::protect)->load();
    return s;
}

void RvcProcessor::timerCallback()
{
    const auto settings = currentSettings();
    if (! isUsingARA())
    {
        streaming.setSettings (settings);  // live mode follows the controls immediately
        return;
    }

    #if JucePlugin_Enable_ARA
    // Offline renders are expensive: wait until the controls settle before re-rendering clips.
    const auto now = juce::Time::getMillisecondCounterHiRes();
    const auto hash = settings.hash();
    if (hash != pendingHash)
    {
        pendingHash = hash;
        pendingSince = now;
    }
    if (hash != committedHash && now - pendingSince >= kCommitDelayMs)
    {
        committedHash = hash;
        committed = settings;
    }
    if (auto* renderer = getPlaybackRenderer<RvcPlaybackRenderer>())
        renderer->update (committed, parameters.getRawParameterValue (ParamIDs::mix)->load(),
                          dbToGain (parameters.getRawParameterValue (ParamIDs::output)->load()));
    #endif
}

RvcProcessor::Status RvcProcessor::getStatus() const
{
    Status s;
    switch (host->getState())
    {
        case EngineHost::State::noFolder: return { Status::Tone::warning, "Choose your models folder", -1.0f };
        case EngineHost::State::loading:  return { Status::Tone::busy, ui ("Loading models…"), -1.0f };
        case EngineHost::State::error:    return { Status::Tone::warning, host->getStatusText(), -1.0f };
        case EngineHost::State::ready:    break;
    }
    if (getVoicePath (Slot::A).isEmpty() && getVoicePath (Slot::B).isEmpty())
        return { Status::Tone::idle, "Pick a voice", -1.0f };

    #if JucePlugin_Enable_ARA
    if (isUsingARA())
    {
        auto* renderer = getPlaybackRenderer<RvcPlaybackRenderer>();
        if (renderer == nullptr)
            return { Status::Tone::idle, ui ("ARA · no clips"), -1.0f };
        const auto r = renderer->getStatus();
        if (r.error.isNotEmpty())
            return { Status::Tone::warning, r.error, -1.0f };
        if (r.clips == 0)
            return { Status::Tone::idle, ui ("ARA · add clips to this track"), -1.0f };
        if (r.ready < r.clips || committedHash != pendingHash)
            return { Status::Tone::busy, "Rendering " + juce::String (juce::roundToInt (r.progress * 100.0f)) + " %", r.progress };
        return { Status::Tone::ok, ui ("ARA · ") + juce::String (r.clips) + (r.clips == 1 ? " clip" : " clips") + " ready", -1.0f };
    }
    #endif

    const float load = streaming.getLoad();
    const auto latency = juce::String (streaming.getLatencySamples() / juce::jmax (1.0, getSampleRate()), 2);
    if (load > 1.0f)
        return { Status::Tone::warning, ui ("Live · too slow for real time (") + juce::String (load, 1) + ui ("×)"), -1.0f };
    return { Status::Tone::ok, ui ("Live · ") + latency + " s latency", -1.0f };
}

//==============================================================================================================
void RvcProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = parameters.copyState().createXml())
        copyXmlToBinary (*xml, destData);
}

void RvcProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
        if (xml->hasTagName (parameters.state.getType()))
            parameters.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessorEditor* RvcProcessor::createEditor() { return new RvcEditor (*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new RvcProcessor(); }
