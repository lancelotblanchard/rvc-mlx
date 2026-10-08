#pragma once

#include <JuceHeader.h>

#include "LookAndFeel.h"
#include "PluginProcessor.h"

/** A labelled rotary control bound to a parameter. */
class Knob : public juce::Component
{
public:
    Knob (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, const juce::String& title, juce::Colour accent);
    void resized() override;
    void paint (juce::Graphics& g) override;

private:
    juce::Slider slider;
    juce::Label value;
    juce::String title;
    juce::AudioProcessorValueTreeState::SliderAttachment attachment;
};

/** Voice picker card (A or B). */
class VoiceCard : public juce::Component
{
public:
    VoiceCard (const juce::String& title, juce::Colour accent);
    void setVoices (const juce::Array<VoiceEntry>& voices, const juce::String& selectedPath, bool allowNone);
    std::function<void (const juce::String& path)> onChange;
    void paint (juce::Graphics& g) override;
    void resized() override;

private:
    juce::String title, detail;
    juce::Colour accent;
    juce::ComboBox box;
    juce::Array<VoiceEntry> voices;
};

class RvcEditor : public juce::AudioProcessorEditor,
                  #if JucePlugin_Enable_ARA
                  public juce::AudioProcessorEditorARAExtension,
                  #endif
                  private juce::ChangeListener,
                  private juce::Timer
{
public:
    explicit RvcEditor (RvcProcessor&);
    ~RvcEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void changeListenerCallback (juce::ChangeBroadcaster*) override;
    void timerCallback() override;
    void refreshVoices();
    void refreshBlendState();
    void showSettingsMenu();

    RvcProcessor& processor;
    RvcLookAndFeel lnf;

    VoiceCard cardA { "VOICE A", Palette::voiceA }, cardB { "VOICE B", Palette::voiceB };
    juce::Slider blend;
    juce::AudioProcessorValueTreeState::SliderAttachment blendAttachment;
    juce::String blendHint;

    Knob pitch, index, protect, mix, output;
    juce::TextButton settingsButton { "Settings" };
    std::unique_ptr<juce::FileChooser> chooser;

    RvcProcessor::Status status;
    juce::Rectangle<int> statusArea, blendArea;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RvcEditor)
};
