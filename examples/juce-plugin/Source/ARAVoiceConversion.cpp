#include "ARAVoiceConversion.h"

#if JucePlugin_Enable_ARA

//==============================================================================================================
RenderCache::RenderCache (EngineHost& h) : host (h) {}

RenderCache::~RenderCache()
{
    for (auto& [_, clip] : clips)
        clip->cancelled = true;
    for (auto* source : listening)
        source->removeListener (this);
}

std::shared_ptr<const RenderedClip> RenderCache::request (juce::ARAAudioSource* source, const RenderSettings& settings,
                                                          double playbackRate)
{
    const Key key { source, settings.hash() * 31 + juce::roundToInt (playbackRate) };
    if (auto it = clips.find (key); it != clips.end())
        return it->second;

    if (listening.insert (source).second)
        source->addListener (this);

    auto clip = std::make_shared<RenderedClip>();
    clips[key] = clip;

    // The reader is created and destroyed on the message thread; only reading happens on the engine thread.
    auto reader = std::make_shared<juce::ARAAudioSourceReader> (source);
    host.submit ([clip, reader, settings, playbackRate, &h = host] (rvc::Engine& engine) mutable
    {
        auto releaseReader = [&reader]
        {
            juce::MessageManager::callAsync ([r = std::move (reader)] {});
        };
        auto fail = [&] (const juce::String& message)
        {
            clip->error = message;
            clip->state = RenderedClip::failed;
            releaseReader();
        };
        if (clip->cancelled)
            return releaseReader();

        clip->state = RenderedClip::reading;
        const auto length = (int) reader->lengthInSamples;
        const int channels = (int) reader->numChannels;
        const double sourceRate = reader->sampleRate;
        if (length <= 0 || channels <= 0)
            return fail ("empty audio source");
        juce::AudioBuffer<float> source (channels, length);
        for (int start = 0; start < length; start += 1 << 20)
        {
            const int n = juce::jmin (1 << 20, length - start);
            juce::HeapBlock<float*> dest (channels);
            for (int c = 0; c < channels; ++c)
                dest[c] = source.getWritePointer (c, start);
            if (! reader->read (dest.get(), channels, start, n))
                return fail ("the host didn't provide the clip's samples");
        }
        releaseReader();

        std::vector<float> mono ((size_t) length, 0.0f);
        for (int c = 0; c < channels; ++c)
            juce::FloatVectorOperations::addWithMultiply (mono.data(), source.getReadPointer (c), 1.0f / (float) channels, length);

        // Store the dry signal at the playback rate so it can play (and be mixed) before the conversion finishes.
        if (sourceRate == playbackRate)
        {
            clip->dry = std::move (source);
        }
        else
        {
            std::vector<std::vector<float>> resampled;
            for (int c = 0; c < channels; ++c)
                resampled.push_back (rvc::resample (std::vector<float> (source.getReadPointer (c), source.getReadPointer (c) + length),
                                                    sourceRate, playbackRate));
            clip->dry.setSize (channels, (int) resampled[0].size());
            for (int c = 0; c < channels; ++c)
                clip->dry.copyFrom (c, 0, resampled[(size_t) c].data(), (int) resampled[(size_t) c].size());
        }
        clip->state = RenderedClip::converting;

        try
        {
            auto voice = h.voiceFor (settings);
            if (voice == nullptr)
                return fail ("no voice selected");
            auto converted = engine.convert (rvc::resample (mono, sourceRate, 16000.0), *voice, settings.options(),
                                             [clip] (float p) { clip->progress = p; return ! clip->cancelled.load(); });
            auto wet = rvc::resample (converted, rvc::voiceInfo (*voice).sampleRate, playbackRate);
            wet.resize ((size_t) clip->dry.getNumSamples(), 0.0f);
            clip->wet = std::move (wet);
            clip->progress = 1.0f;
            clip->state = RenderedClip::ready;
        }
        catch (const rvc::Cancelled&)
        {
            clip->state = RenderedClip::failed;
        }
        catch (const std::exception& e)
        {
            fail (e.what());
        }
    });
    return clip;
}

void RenderCache::collectGarbage()
{
    for (auto it = clips.begin(); it != clips.end();)
    {
        if (it->second.use_count() == 1)
        {
            it->second->cancelled = true;
            it = clips.erase (it);
        }
        else
        {
            ++it;
        }
    }
}

void RenderCache::drop (juce::ARAAudioSource* source, bool onlyFailed)
{
    for (auto it = clips.begin(); it != clips.end();)
    {
        if (it->first.first == source && (! onlyFailed || it->second->state.load() == RenderedClip::failed))
        {
            it->second->cancelled = true;
            it = clips.erase (it);
        }
        else
        {
            ++it;
        }
    }
}

void RenderCache::doUpdateAudioSourceContent (juce::ARAAudioSource* source, juce::ARAContentUpdateScopes scopeFlags)
{
    if (scopeFlags.affectSamples())
        drop (source);  // renderers re-request on their next update
}

void RenderCache::didEnableAudioSourceSamplesAccess (juce::ARAAudioSource* source, bool enable)
{
    if (enable)
        drop (source, true);  // retry clips that failed because the host hadn't granted access yet
}

void RenderCache::willDestroyAudioSource (juce::ARAAudioSource* source)
{
    drop (source);
    source->removeListener (this);
    listening.erase (source);
}

//==============================================================================================================
void RvcPlaybackRenderer::prepareToPlay (double sampleRate, int maximumSamplesPerBlock, int numChannels,
                                         juce::AudioProcessor::ProcessingPrecision precision, AlwaysNonRealtime alwaysNonRealtime)
{
    juce::ARAPlaybackRenderer::prepareToPlay (sampleRate, maximumSamplesPerBlock, numChannels, precision, alwaysNonRealtime);
    playbackRate = sampleRate;
}

void RvcPlaybackRenderer::update (const RenderSettings& settings, float newMix, float newGain)
{
    mix = newMix;
    gain = newGain;
    const double rate = playbackRate.load();
    auto* dc = juce::ARADocumentControllerSpecialisation::getSpecialisedDocumentController<RvcDocumentController> (getDocumentController());
    if (rate <= 0.0 || dc == nullptr)
        return;

    auto next = std::make_shared<Table>();
    if (settings.hasVoice())
    {
        for (auto* region : getPlaybackRegions())
        {
            auto* source = region->getAudioModification()->getAudioSource();
            const bool known = std::any_of (next->clips.begin(), next->clips.end(), [source] (const auto& c) { return c.first == source; });
            if (! known)
                next->clips.emplace_back (source, dc->getCache().request (source, settings, rate));
        }
    }
    std::atomic_store (&table, std::shared_ptr<const Table> (std::move (next)));
    dc->getCache().collectGarbage();
}

RvcPlaybackRenderer::Status RvcPlaybackRenderer::getStatus() const
{
    Status s;
    auto t = std::atomic_load (&table);
    if (t == nullptr)
        return s;
    float progress = 0.0f;
    for (const auto& [_, clip] : t->clips)
    {
        ++s.clips;
        const int state = clip->state.load();
        if (state == RenderedClip::ready)
            ++s.ready;
        if (state == RenderedClip::failed && clip->error.isNotEmpty())
            s.error = clip->error;
        progress += state == RenderedClip::ready ? 1.0f : clip->progress.load();
    }
    s.progress = s.clips > 0 ? progress / (float) s.clips : 1.0f;
    return s;
}

bool RvcPlaybackRenderer::processBlock (juce::AudioBuffer<float>& buffer, juce::AudioProcessor::Realtime,
                                        const juce::AudioPlayHead::PositionInfo& positionInfo) noexcept
{
    const int numSamples = buffer.getNumSamples();
    buffer.clear();
    if (! positionInfo.getIsPlaying())
        return true;

    const auto t = std::atomic_load (&table);
    if (t == nullptr)
        return true;

    const double rate = playbackRate.load();
    const float wetAmount = mix.load(), outGain = gain.load();
    const auto blockStart = positionInfo.getTimeInSamples().orFallback (0);
    const juce::Range<juce::int64> blockRange { blockStart, blockStart + numSamples };

    for (auto* region : getPlaybackRegions())
    {
        const auto* source = region->getAudioModification()->getAudioSource();
        const RenderedClip* clip = nullptr;
        for (const auto& [s, c] : t->clips)
            if (s == source) { clip = c.get(); break; }
        if (clip == nullptr || clip->state.load (std::memory_order_acquire) < RenderedClip::converting)
            continue;  // still reading the source: silence for a moment

        const auto regionRange = region->getSampleRange (rate);
        const auto renderRange = blockRange.getIntersectionWith (regionRange);
        if (renderRange.isEmpty())
            continue;

        // Playback position -> position in the (playback-rate) rendered source.
        const auto modificationStart = (juce::int64) std::llround (region->getStartInAudioModificationTime() * rate);
        const auto sourceOffset = modificationStart - regionRange.getStart();
        const bool wetReady = clip->state.load (std::memory_order_acquire) == RenderedClip::ready;
        const int dryChannels = clip->dry.getNumChannels(), length = clip->dry.getNumSamples();

        for (auto pos = renderRange.getStart(); pos < renderRange.getEnd(); ++pos)
        {
            const auto src = pos + sourceOffset;
            if (src < 0 || src >= length)
                continue;
            const int i = (int) (pos - blockStart);
            const float wet = wetReady ? clip->wet[(size_t) src] : 0.0f;
            for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            {
                const float dry = clip->dry.getSample (juce::jmin (ch, dryChannels - 1), (int) src);
                const float v = wetReady ? wetAmount * wet + (1.0f - wetAmount) * dry : dry;
                buffer.addSample (ch, i, v * outGain);
            }
        }
    }
    return true;
}

//==============================================================================================================
juce::ARAPlaybackRenderer* RvcDocumentController::doCreatePlaybackRenderer() noexcept
{
    return new RvcPlaybackRenderer (getDocumentController());
}

// Settings live in each plug-in instance's state (see RvcProcessor::getStateInformation), and rendered audio is
// recomputed on load, so the ARA archive carries no data of its own.
bool RvcDocumentController::doRestoreObjectsFromStream (juce::ARAInputStream&, const juce::ARARestoreObjectsFilter*) noexcept
{
    return true;
}

bool RvcDocumentController::doStoreObjectsToStream (juce::ARAOutputStream&, const juce::ARAStoreObjectsFilter*) noexcept
{
    return true;
}

const ARA::ARAFactory* JUCE_CALLTYPE createARAFactory()
{
    return juce::ARADocumentControllerSpecialisation::createARAFactory<RvcDocumentController>();
}

#endif
