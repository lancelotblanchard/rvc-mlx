// Headless smoke test for the plug-in's live (non-ARA) path and editor.
//
//   RVCMorphSmokeTest <models folder> <voice A> [voice B] [snapshot.png]
//
// Streams a synthetic voice-like signal through RvcProcessor at 48 kHz in 512-sample blocks (paced like a real
// audio callback so the engine thread has time to work), then checks that converted audio arrives after the reported
// latency, is finite, non-silent and free of underruns. Optionally renders the editor to a PNG.
#include <JuceHeader.h>

#include "../Source/EngineHost.h"
#include "../Source/PluginEditor.h"
#include "../Source/PluginProcessor.h"

int main (int argc, char** argv)
{
    juce::ScopedJuceInitialiser_GUI gui;
    if (argc < 3)
    {
        std::printf ("usage: %s <models> <voiceA> [voiceB] [snapshot.png]\n", argv[0]);
        return 2;
    }
    const juce::File models (argv[1]), voiceA (argv[2]);
    const juce::String voiceB = argc > 3 ? juce::String (argv[3]) : juce::String();
    int failures = 0;
    auto check = [&] (bool ok, const juce::String& what) { std::printf ("%s  %s\n", ok ? "ok  " : "FAIL", what.toRawUTF8()); failures += ok ? 0 : 1; };

    juce::SharedResourcePointer<EngineHost> host;
    host->setModelsFolder (models);
    for (int i = 0; i < 600 && host->getState() == EngineHost::State::loading; ++i)
        juce::MessageManager::getInstance()->runDispatchLoopUntil (100);
    check (host->getState() == EngineHost::State::ready, "engine ready: " + host->getStatusText());
    check (host->getVoices().size() > 0, juce::String (host->getVoices().size()) + " voices found");

    RvcProcessor processor;
    processor.setVoicePath (RvcProcessor::Slot::A, voiceA.getFullPathName());
    if (voiceB.isNotEmpty())
    {
        processor.setVoicePath (RvcProcessor::Slot::B, voiceB);
        processor.getParameters().getParameter (ParamIDs::blend)->setValueNotifyingHost (0.5f);
    }
    processor.getParameters().getParameter (ParamIDs::pitch)->setValueNotifyingHost (
        processor.getParameters().getParameter (ParamIDs::pitch)->convertTo0to1 (5.0f));

    const double sr = 48000.0;
    const int block = 512;
    processor.setPlayConfigDetails (2, 2, sr, block);
    processor.prepareToPlay (sr, block);
    const int latency = processor.getLatencySamples();
    check (latency > 0 && latency < (int) sr, "reported latency " + juce::String (latency / sr, 3) + " s");
    juce::MessageManager::getInstance()->runDispatchLoopUntil (300);  // let the settings timer push the voice

    if (argc > 4)  // editor snapshot, taken before streaming starts
    {
        std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
        juce::MessageManager::getInstance()->runDispatchLoopUntil (300);
        auto image = editor->createComponentSnapshot (editor->getLocalBounds(), true, 2.0f);
        juce::File png (argv[4]);
        png.deleteFile();
        juce::FileOutputStream stream (png);
        juce::PNGImageFormat().writeImageToStream (image, stream);
        check (png.existsAsFile(), "editor snapshot -> " + png.getFullPathName());
    }

    // A vowel-like test signal: glottal-ish harmonics with vibrato, 6 seconds.
    const int total = (int) (6.0 * sr);
    std::vector<float> outL;
    juce::AudioBuffer<float> buffer (2, block);
    juce::MidiBuffer midi;
    double phase = 0;
    for (int pos = 0; pos < total; pos += block)
    {
        for (int i = 0; i < block; ++i)
        {
            const double t = (pos + i) / sr;
            phase += 2.0 * juce::MathConstants<double>::pi * (160.0 + 12.0 * std::sin (2 * juce::MathConstants<double>::pi * 5 * t)) / sr;
            float v = 0;
            for (int h = 1; h <= 12; ++h)
                v += (float) (std::sin (h * phase) / h);
            buffer.setSample (0, i, 0.2f * v);
            buffer.setSample (1, i, 0.2f * v);
        }
        processor.processBlock (buffer, midi);
        outL.insert (outL.end(), buffer.getReadPointer (0), buffer.getReadPointer (0) + block);
        // Pace like a real-time callback, and keep the message loop alive for the settings timer.
        juce::MessageManager::getInstance()->runDispatchLoopUntil ((int) (1000.0 * block / sr));
    }

    auto rms = [&] (int from, int to)
    {
        double acc = 0;
        for (int i = from; i < to; ++i) acc += (double) outL[(size_t) i] * outL[(size_t) i];
        return std::sqrt (acc / juce::jmax (1, to - from));
    };
    bool finite = std::all_of (outL.begin(), outL.end(), [] (float v) { return std::isfinite (v); });
    check (finite, "output is finite");
    const double before = rms (0, juce::jmax (1, latency - block)), after = rms (latency + (int) sr, total);
    check (before < 1e-4, "silence before the latency (rms " + juce::String (before, 6) + ")");
    check (after > 1e-3, "converted audio after the latency (rms " + juce::String (after, 4) + ")");
    // Real-time speed depends on the machine (CPU-only MLX builds are far slower than Metal), so it is reported only.
    const auto status = processor.getStatus();
    const bool slow = status.text.contains ("too slow");
    check (status.tone != RvcProcessor::Status::Tone::warning || slow,
           "status: " + status.text + (slow ? " (machine-dependent, informational)" : ""));

    processor.releaseResources();
    std::printf ("\n%s\n", failures == 0 ? "all checks passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
