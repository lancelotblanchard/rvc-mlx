#include "StreamingConverter.h"

#include "EngineHost.h"

StreamingConverter::StreamingConverter (EngineHost& h) : host (h) {}

StreamingConverter::~StreamingConverter() { release(); }

void StreamingConverter::prepare (double newSampleRate, int maximumBlockSize)
{
    release();  // waits for any pump in progress

    sampleRate = newSampleRate;
    hop = juce::roundToInt (kHopSeconds * sampleRate);
    crossfade = juce::roundToInt (kCrossfadeSeconds * sampleRate);
    search = juce::roundToInt (kSearchSeconds * sampleRate);
    context = juce::roundToInt (kContextSeconds * sampleRate);
    margin = 64;  // resampler edge, discarded on both sides of each converted chunk

    // The engine thread gets one hop of slack; the SOLA alignment adds the crossfade and half the search range.
    const int prefill = 2 * hop;
    latency = prefill + crossfade + search / 2;

    inFifo.setTotalSize (4 * hop + maximumBlockSize + 1);
    outFifo.setTotalSize (prefill + 4 * hop + maximumBlockSize + 1);
    inBuffer.assign ((size_t) inFifo.getTotalSize(), 0.0f);
    outBuffer.assign ((size_t) outFifo.getTotalSize(), 0.0f);
    inFifo.reset();
    outFifo.reset();
    {
        int s1, n1, s2, n2;
        outFifo.prepareToWrite (prefill, s1, n1, s2, n2);
        outFifo.finishedWrite (n1 + n2);  // buffer already zeroed
    }

    history.assign ((size_t) (context + hop), 0.0f);
    hopIn.assign ((size_t) hop, 0.0f);
    hopOut.assign ((size_t) hop, 0.0f);
    previousTail.assign ((size_t) crossfade, 0.0f);
    havePreviousTail = false;
    fade.resize ((size_t) crossfade);
    for (int i = 0; i < crossfade; ++i)
        fade[(size_t) i] = std::pow (std::sin (juce::MathConstants<float>::halfPi * (float) (i + 0.5f) / (float) crossfade), 2.0f);

    underruns = 0;
    prepared = true;
    host.addStreamClient (this);
}

void StreamingConverter::release()
{
    host.removeStreamClient (this);
    prepared = false;
}

void StreamingConverter::setSettings (const RenderSettings& s)
{
    const juce::SpinLock::ScopedLockType sl (settingsLock);
    settings = s;
}

void StreamingConverter::process (const float* input, float* output, int numSamples) noexcept
{
    if (! prepared.load())
    {
        juce::FloatVectorOperations::clear (output, numSamples);
        return;
    }

    int s1, n1, s2, n2;
    inFifo.prepareToWrite (numSamples, s1, n1, s2, n2);  // on overflow the newest samples are dropped
    juce::FloatVectorOperations::copy (inBuffer.data() + s1, input, n1);
    juce::FloatVectorOperations::copy (inBuffer.data() + s2, input + n1, n2);
    inFifo.finishedWrite (n1 + n2);
    if (inFifo.getNumReady() >= hop)
        host.wakeUp();

    outFifo.prepareToRead (numSamples, s1, n1, s2, n2);
    juce::FloatVectorOperations::copy (output, outBuffer.data() + s1, n1);
    juce::FloatVectorOperations::copy (output + n1, outBuffer.data() + s2, n2);
    outFifo.finishedRead (n1 + n2);
    if (n1 + n2 < numSamples)
    {
        juce::FloatVectorOperations::clear (output + n1 + n2, numSamples - n1 - n2);
        underruns.fetch_add (1);
    }
}

void StreamingConverter::pump (rvc::Engine& engine, EngineHost& h)
{
    if (! prepared.load())
        return;
    while (inFifo.getNumReady() >= hop && outFifo.getFreeSpace() >= hop)
    {
        int s1, n1, s2, n2;
        inFifo.prepareToRead (hop, s1, n1, s2, n2);
        std::copy (inBuffer.begin() + s1, inBuffer.begin() + s1 + n1, hopIn.begin());
        std::copy (inBuffer.begin() + s2, inBuffer.begin() + s2 + n2, hopIn.begin() + n1);
        inFifo.finishedRead (n1 + n2);

        const auto start = juce::Time::getMillisecondCounterHiRes();
        convertHop (engine, h, hopOut.data());
        const auto elapsed = (juce::Time::getMillisecondCounterHiRes() - start) / 1000.0;
        load = (float) (elapsed / kHopSeconds);

        outFifo.prepareToWrite (hop, s1, n1, s2, n2);
        std::copy (hopOut.begin(), hopOut.begin() + n1, outBuffer.begin() + s1);
        std::copy (hopOut.begin() + n1, hopOut.begin() + n1 + n2, outBuffer.begin() + s2);
        outFifo.finishedWrite (n1 + n2);
    }
}

void StreamingConverter::convertHop (rvc::Engine& engine, EngineHost& h, float* out)
{
    // Slide the history window and append the new hop.
    std::move (history.begin() + hop, history.end(), history.begin());
    std::copy (hopIn.begin(), hopIn.end(), history.end() - hop);

    RenderSettings s;
    {
        const juce::SpinLock::ScopedLockType sl (settingsLock);
        s = settings;
    }

    // `chunk` is the last (hop + crossfade + search) samples of the converted window, at the host rate.
    const int needed = hop + crossfade + search;
    std::vector<float> chunk;
    rvc::VoicePtr voice;
    try { voice = h.voiceFor (s); }
    catch (const std::exception&) { voice = nullptr; }

    if (voice != nullptr)
    {
        try
        {
            const int voiceRate = rvc::voiceInfo (*voice).sampleRate;
            const auto window16k = rvc::resample (history, sampleRate, 16000.0);
            const int voiceSamples = (int) std::ceil ((needed + 2 * margin) * voiceRate / sampleRate) + 1;
            const auto converted = engine.convertWindow (window16k, *voice, s.options(), voiceSamples);
            const auto atHostRate = rvc::resample (converted, voiceRate, sampleRate);
            const int end = (int) atHostRate.size() - margin;
            if (end - needed >= 0)
                chunk.assign (atHostRate.begin() + (end - needed), atHostRate.begin() + end);
        }
        catch (const std::exception& e)
        {
            DBG ("live conversion failed: " << e.what());
        }
    }
    if (chunk.empty())  // no voice yet (or an error): pass the input through with the same timing
        chunk.assign (history.end() - margin - needed, history.end() - margin);

    // SOLA: best alignment of the new chunk against the previous tail, then an equal-power crossfade.
    int offset = 0;
    if (havePreviousTail)
    {
        double best = -1e30;
        for (int o = 0; o <= search; ++o)
        {
            double num = 0, energy = 1e-9;
            for (int i = 0; i < crossfade; ++i)
            {
                num += (double) chunk[(size_t) (o + i)] * previousTail[(size_t) i];
                energy += (double) chunk[(size_t) (o + i)] * chunk[(size_t) (o + i)];
            }
            const double score = num / std::sqrt (energy);
            if (score > best) { best = score; offset = o; }
        }
    }
    for (int i = 0; i < hop; ++i)
    {
        float v = chunk[(size_t) (offset + i)];
        if (i < crossfade && havePreviousTail)
            v = previousTail[(size_t) i] * (1.0f - fade[(size_t) i]) + v * fade[(size_t) i];
        out[i] = juce::jlimit (-1.0f, 1.0f, v);
    }
    std::copy (chunk.begin() + offset + hop, chunk.begin() + offset + hop + crossfade, previousTail.begin());
    havePreviousTail = true;
}
