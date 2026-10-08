#include "EngineHost.h"

#include "StreamingConverter.h"

namespace
{
juce::PropertiesFile::Options propertiesOptions()
{
    juce::PropertiesFile::Options o;
    o.applicationName = "RVC Morph";
    o.filenameSuffix = ".settings";
    o.folderName = "RVC Morph";
    o.osxLibrarySubFolder = "Application Support";
    return o;
}

juce::File defaultModelsFolder()
{
    return juce::File::getSpecialLocation (juce::File::userMusicDirectory).getChildFile ("RVC Morph").getChildFile ("models");
}
} // namespace

//==============================================================================================================
class EngineHost::Worker : public juce::Thread
{
public:
    explicit Worker (EngineHost& h) : juce::Thread ("rvc-mlx engine"), host (h) {}

    void run() override
    {
        while (! threadShouldExit())
        {
            host.wakeEvent.wait (20);
            if (threadShouldExit())
                break;

            ensureEngine();
            if (host.engine == nullptr)
                continue;

            pumpStreams();

            Job job;
            {
                const juce::ScopedLock sl (host.lock);
                if (! host.jobs.empty())
                {
                    job = std::move (host.jobs.front());
                    host.jobs.pop_front();
                }
            }
            if (job)
            {
                try { job (*host.engine); }
                catch (const std::exception& e) { DBG ("rvc job failed: " << e.what()); }
            }
        }
        host.voiceCache.clear();
        host.blendCache.clear();
        host.engine.reset();
    }

private:
    void ensureEngine()
    {
        juce::File folder;
        {
            const juce::ScopedLock sl (host.lock);
            folder = host.modelsFolder;
        }
        if (folder == host.engineFolder)
            return;

        host.engine.reset();
        host.voiceCache.clear();
        host.blendCache.clear();
        host.engineFolder = folder;
        if (folder == juce::File())
            return;

        setState (State::loading, ui ("Loading models…"));
        try
        {
            host.engine = rvc::Engine::fromFolder (folder.getFullPathName().toStdString());
            setState (State::ready, "Ready");
        }
        catch (const std::exception& e)
        {
            setState (State::error, e.what());
        }
    }

    void pumpStreams()
    {
        const juce::ScopedLock sl (host.streamLock);  // clients unregister under this lock, so they can't vanish mid-pump
        for (auto* client : host.streamClients)
            client->pump (*host.engine, host);
    }

    void setState (State s, const juce::String& text)
    {
        {
            const juce::ScopedLock sl (host.lock);
            host.state = s;
            host.statusText = text;
        }
        host.sendChangeMessage();
    }

    EngineHost& host;
};

//==============================================================================================================
EngineHost::EngineHost()
    : properties (std::make_unique<juce::PropertiesFile> (propertiesOptions()))
{
    worker = std::make_unique<Worker> (*this);
    worker->startThread (juce::Thread::Priority::high);

    auto saved = properties->getValue ("modelsFolder");
    auto folder = saved.isNotEmpty() ? juce::File (saved) : defaultModelsFolder();
    if (folder.isDirectory())
        setModelsFolder (folder);
}

EngineHost::~EngineHost()
{
    worker->signalThreadShouldExit();
    wakeEvent.signal();
    worker->stopThread (10000);
}

void EngineHost::setModelsFolder (const juce::File& folder)
{
    {
        const juce::ScopedLock sl (lock);
        modelsFolder = folder;
        state = folder.isDirectory() ? State::loading : State::noFolder;
        statusText = folder.isDirectory() ? ui ("Loading models…") : "Choose your models folder";
        properties->setValue ("modelsFolder", folder.getFullPathName());
        properties->saveIfNeeded();
    }
    rescanVoices();
    wakeEvent.signal();
}

juce::File EngineHost::getModelsFolder() const
{
    const juce::ScopedLock sl (lock);
    return modelsFolder;
}

EngineHost::State EngineHost::getState() const
{
    const juce::ScopedLock sl (lock);
    return state;
}

juce::String EngineHost::getStatusText() const
{
    const juce::ScopedLock sl (lock);
    return statusText;
}

juce::Array<VoiceEntry> EngineHost::getVoices() const
{
    const juce::ScopedLock sl (lock);
    return voices;
}

bool EngineHost::readVoiceHeader (const juce::File& file, VoiceEntry& out)
{
    // safetensors: 8-byte little-endian header size, then a JSON header whose "__metadata__" holds our fields.
    juce::FileInputStream in (file);
    if (! in.openedOk() || in.getTotalLength() < 8)
        return false;
    const auto headerSize = (juce::int64) in.readInt64();
    if (headerSize <= 0 || headerSize > 100 * 1024 * 1024)
        return false;
    juce::MemoryBlock block;
    if (in.readIntoMemoryBlock (block, (ssize_t) headerSize) != (size_t) headerSize)
        return false;
    auto json = juce::JSON::parse (block.toString());
    auto meta = json.getProperty ("__metadata__", {});
    if (meta.getProperty ("kind", {}).toString() != "voice")
        return false;
    out.file = file;
    out.name = meta.getProperty ("name", file.getFileNameWithoutExtension()).toString();
    out.version = meta.getProperty ("version", "v2").toString();
    out.mergedFrom = meta.getProperty ("merged_from", {}).toString();
    out.sampleRate = meta.getProperty ("sample_rate", "0").toString().getIntValue();
    out.hasPitch = meta.getProperty ("f0", "1").toString() == "1";
    out.indexSize = meta.getProperty ("index_size", "0").toString().getIntValue();
    return true;
}

void EngineHost::rescanVoices()
{
    juce::Array<VoiceEntry> found;
    const auto dir = getModelsFolder().getChildFile ("voices");
    for (const auto& f : dir.findChildFiles (juce::File::findFiles, false, "*.safetensors"))
    {
        VoiceEntry e;
        if (readVoiceHeader (f, e))
            found.add (e);
    }
    std::sort (found.begin(), found.end(), [] (const VoiceEntry& a, const VoiceEntry& b) { return a.name.compareNatural (b.name) < 0; });
    {
        const juce::ScopedLock sl (lock);
        voices = found;
    }
    sendChangeMessage();
}

juce::String EngineHost::blendProblem (const juce::String& a, const juce::String& b) const
{
    if (a.isEmpty() || b.isEmpty())
        return {};
    VoiceEntry ea, eb;
    if (! readVoiceHeader (juce::File (a), ea) || ! readVoiceHeader (juce::File (b), eb))
        return "missing voice file";
    if (ea.version != eb.version)
        return "can't blend " + ea.version + " with " + eb.version;
    if (ea.sampleRate != eb.sampleRate)
        return "can't blend " + juce::String (ea.sampleRate / 1000) + "k with " + juce::String (eb.sampleRate / 1000) + "k";
    if (ea.hasPitch != eb.hasPitch)
        return "pitch-guided and pitchless voices can't blend";
    return {};
}

void EngineHost::submit (Job job)
{
    {
        const juce::ScopedLock sl (lock);
        jobs.push_back (std::move (job));
    }
    wakeEvent.signal();
}

void EngineHost::addStreamClient (StreamingConverter* c)
{
    const juce::ScopedLock sl (streamLock);
    streamClients.addIfNotAlreadyThere (c);
}

void EngineHost::removeStreamClient (StreamingConverter* c)
{
    const juce::ScopedLock sl (streamLock);
    streamClients.removeFirstMatchingValue (c);
}

rvc::VoicePtr EngineHost::loadCached (const juce::String& path)
{
    auto it = voiceCache.find (path);
    if (it != voiceCache.end())
        return it->second;
    if (voiceCache.size() >= 6)
        voiceCache.erase (voiceCache.begin());
    auto voice = rvc::loadVoice (path.toStdString());
    voiceCache[path] = voice;
    return voice;
}

rvc::VoicePtr EngineHost::voiceFor (const RenderSettings& s)
{
    if (! s.hasVoice())
        return nullptr;
    const int pct = s.blendPercent();
    if (s.voiceB.isEmpty() || pct == 0)
        return loadCached (s.voiceA.isNotEmpty() ? s.voiceA : s.voiceB);
    if (s.voiceA.isEmpty() || pct == 100)
        return loadCached (s.voiceB);

    auto a = loadCached (s.voiceA), b = loadCached (s.voiceB);
    if (! rvc::blendIncompatibility (*a, *b).empty())
        return pct < 50 ? a : b;  // the editor explains why blending is disabled

    const juce::String key = s.voiceA + "|" + s.voiceB + "|" + juce::String (pct);
    auto it = blendCache.find (key);
    if (it != blendCache.end())
        return it->second;
    if (blendCache.size() >= 4)
        blendCache.erase (blendCache.begin());
    auto blended = rvc::blendVoices ({ a, b }, { 100.0f - (float) pct, (float) pct });
    blendCache[key] = blended;
    return blended;
}
