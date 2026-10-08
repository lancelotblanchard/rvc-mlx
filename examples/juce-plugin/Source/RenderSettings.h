#pragma once

#include <JuceHeader.h>

#include <rvc/rvc.h>

/** UI text from a UTF-8 literal (juce::String treats plain `const char*` as ASCII). */
inline juce::String ui (const char* utf8) { return juce::String::fromUTF8 (utf8); }

/** Everything that changes the converted audio (mix and output gain are applied at playback instead). */
struct RenderSettings
{
    juce::String voiceA, voiceB;  // absolute paths of converted voices; empty = slot unused
    float blend = 0.0f;           // 0 = all A, 1 = all B
    float pitch = 0.0f;           // semitones
    float indexRate = 0.75f;
    float protect = 0.33f;

    bool hasVoice() const { return voiceA.isNotEmpty() || voiceB.isNotEmpty(); }

    /** Blend is quantised to 1 % steps so the voice cache stays small while dragging. */
    int blendPercent() const { return juce::roundToInt (juce::jlimit (0.0f, 1.0f, blend) * 100.0f); }

    juce::int64 hash() const
    {
        juce::String s;
        s << voiceA << '|' << voiceB << '|' << blendPercent() << '|' << juce::roundToInt (pitch * 100.0f) << '|'
          << juce::roundToInt (indexRate * 1000.0f) << '|' << juce::roundToInt (protect * 1000.0f);
        return s.hashCode64();
    }

    rvc::ConvertOptions options() const
    {
        rvc::ConvertOptions o;
        o.pitch = pitch;
        o.indexRate = indexRate;
        o.protect = protect;
        return o;
    }
};
