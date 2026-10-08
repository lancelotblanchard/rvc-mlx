#include "PluginEditor.h"

namespace
{
constexpr int kWidth = 680, kHeight = 440, kPad = 20;

juce::String describe (const VoiceEntry& v)
{
    juce::String s = v.version + ui (" · ") + juce::String (v.sampleRate / 1000) + " kHz";
    if (v.indexSize > 0)
        s << ui (" · index");
    if (! v.hasPitch)
        s << ui (" · no pitch");
    if (v.mergedFrom.isNotEmpty())
        s << ui (" · blend");
    return s;
}
} // namespace

//==============================================================================================================
Knob::Knob (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, const juce::String& t, juce::Colour accent)
    : title (t), attachment (state, paramId, slider)
{
    slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    slider.setColour (juce::Slider::rotarySliderFillColourId, accent);
    slider.setRotaryParameters (juce::MathConstants<float>::pi * 1.25f, juce::MathConstants<float>::pi * 2.75f, true);
    slider.setDoubleClickReturnValue (true, state.getParameter (paramId)->convertFrom0to1 (state.getParameter (paramId)->getDefaultValue()));
    addAndMakeVisible (slider);

    value.setJustificationType (juce::Justification::centred);
    value.setFont (uiFont (13.0f));
    value.setColour (juce::Label::textColourId, Palette::muted);
    value.setInterceptsMouseClicks (false, false);
    addAndMakeVisible (value);

    auto* param = state.getParameter (paramId);
    auto update = [this, param] { value.setText (param->getCurrentValueAsText(), juce::dontSendNotification); };
    slider.onValueChange = update;
    update();
}

void Knob::resized()
{
    auto r = getLocalBounds();
    r.removeFromTop (20);
    value.setBounds (r.removeFromBottom (18));
    slider.setBounds (r);
}

void Knob::paint (juce::Graphics& g)
{
    g.setColour (Palette::text);
    g.setFont (uiFont (13.0f, true));
    g.drawText (title, getLocalBounds().removeFromTop (18), juce::Justification::centred);
}

//==============================================================================================================
VoiceCard::VoiceCard (const juce::String& t, juce::Colour c) : title (t), accent (c)
{
    box.setTextWhenNothingSelected ("Choose a voice");
    box.setTextWhenNoChoicesAvailable ("No voices in models/voices");
    box.onChange = [this]
    {
        const int i = box.getSelectedId() - 2;
        const auto path = juce::isPositiveAndBelow (i, voices.size()) ? voices[i].file.getFullPathName() : juce::String();
        detail = juce::isPositiveAndBelow (i, voices.size()) ? describe (voices[i]) : juce::String();
        repaint();
        if (onChange)
            onChange (path);
    };
    addAndMakeVisible (box);
}

void VoiceCard::setVoices (const juce::Array<VoiceEntry>& v, const juce::String& selectedPath, bool allowNone)
{
    voices = v;
    box.clear (juce::dontSendNotification);
    if (allowNone)
        box.addItem ("None", 1);
    int selected = 0;
    for (int i = 0; i < voices.size(); ++i)
    {
        box.addItem (voices[i].name, i + 2);
        if (voices[i].file.getFullPathName() == selectedPath)
            selected = i + 2;
    }
    if (selected == 0 && allowNone && selectedPath.isEmpty())
        selected = 1;
    box.setSelectedId (selected, juce::dontSendNotification);
    detail = selected >= 2 ? describe (voices[selected - 2]) : (selectedPath.isNotEmpty() ? "missing: " + juce::File (selectedPath).getFileName() : "");
    repaint();
}

void VoiceCard::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat();
    g.setColour (Palette::card);
    g.fillRoundedRectangle (r, 14.0f);
    g.setColour (Palette::cardEdge);
    g.drawRoundedRectangle (r.reduced (0.5f), 14.0f, 1.0f);

    auto text = getLocalBounds().reduced (16, 14);
    auto top = text.removeFromTop (16);
    g.setColour (accent);
    g.fillEllipse (top.removeFromLeft (8).withSizeKeepingCentre (8, 8).toFloat());
    top.removeFromLeft (8);
    g.setFont (uiFont (11.5f, true));
    g.drawText (title, top, juce::Justification::centredLeft);

    g.setColour (Palette::muted);
    g.setFont (uiFont (12.5f));
    g.drawText (detail, text.removeFromBottom (16), juce::Justification::centredLeft);
}

void VoiceCard::resized()
{
    box.setBounds (getLocalBounds().reduced (12, 0).withTop (38).withHeight (34));
}

//==============================================================================================================
RvcEditor::RvcEditor (RvcProcessor& p)
    : AudioProcessorEditor (&p),
      #if JucePlugin_Enable_ARA
      AudioProcessorEditorARAExtension (&p),
      #endif
      processor (p),
      blendAttachment (p.getParameters(), ParamIDs::blend, blend),
      pitch (p.getParameters(), ParamIDs::pitch, "Pitch", Palette::voiceA),
      index (p.getParameters(), ParamIDs::index, "Index", Palette::voiceA),
      protect (p.getParameters(), ParamIDs::protect, "Protect", Palette::voiceA),
      mix (p.getParameters(), ParamIDs::mix, "Mix", Palette::voiceB),
      output (p.getParameters(), ParamIDs::output, "Output", Palette::voiceB)
{
    setLookAndFeel (&lnf);
    for (auto* c : std::initializer_list<juce::Component*> { &cardA, &cardB, &blend, &pitch, &index, &protect, &mix, &output, &settingsButton })
        addAndMakeVisible (c);

    blend.setSliderStyle (juce::Slider::LinearHorizontal);
    blend.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    blend.onValueChange = [this] { repaint (blendArea); };

    cardA.onChange = [this] (const juce::String& path) { processor.setVoicePath (RvcProcessor::Slot::A, path); refreshBlendState(); };
    cardB.onChange = [this] (const juce::String& path) { processor.setVoicePath (RvcProcessor::Slot::B, path); refreshBlendState(); };

    settingsButton.onClick = [this] { showSettingsMenu(); };
    processor.getEngineHost().addChangeListener (this);
    refreshVoices();

    setSize (kWidth, kHeight);
    startTimerHz (10);
}

RvcEditor::~RvcEditor()
{
    processor.getEngineHost().removeChangeListener (this);
    setLookAndFeel (nullptr);
}

void RvcEditor::refreshVoices()
{
    const auto voices = processor.getEngineHost().getVoices();
    cardA.setVoices (voices, processor.getVoicePath (RvcProcessor::Slot::A), false);
    cardB.setVoices (voices, processor.getVoicePath (RvcProcessor::Slot::B), true);
    refreshBlendState();
}

void RvcEditor::refreshBlendState()
{
    const auto a = processor.getVoicePath (RvcProcessor::Slot::A), b = processor.getVoicePath (RvcProcessor::Slot::B);
    if (b.isEmpty() || a.isEmpty())
        blendHint = "Pick a second voice to morph between them";
    else
        blendHint = processor.getEngineHost().blendProblem (a, b);
    blend.setEnabled (a.isNotEmpty() && b.isNotEmpty() && blendHint.isEmpty());
    repaint();
}

void RvcEditor::changeListenerCallback (juce::ChangeBroadcaster*) { refreshVoices(); }

void RvcEditor::timerCallback()
{
    status = processor.getStatus();
    repaint (statusArea);
}

void RvcEditor::showSettingsMenu()
{
    auto& host = processor.getEngineHost();
    const auto folder = host.getModelsFolder();
    juce::PopupMenu menu;
    menu.addSectionHeader ("Models folder");
    menu.addItem (100, folder == juce::File() ? juce::String ("(not set)") : folder.getFullPathName(), false);
    menu.addItem (1, ui ("Choose models folder…"));
    menu.addItem (2, "Reveal in Finder", folder.isDirectory());
    menu.addItem (3, "Rescan voices", folder.isDirectory());
    menu.addSeparator();
    menu.addItem (4, ui ("How to convert voices…"));
    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (settingsButton), [this, folder] (int result)
    {
        auto& h = processor.getEngineHost();
        if (result == 1)
        {
            chooser = std::make_unique<juce::FileChooser> ("Choose the folder with hubert.safetensors, rmvpe.safetensors and voices/",
                                                           folder.isDirectory() ? folder : juce::File::getSpecialLocation (juce::File::userMusicDirectory));
            chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
                                  [this] (const juce::FileChooser& fc)
                                  {
                                      if (fc.getResult().isDirectory())
                                          processor.getEngineHost().setModelsFolder (fc.getResult());
                                  });
        }
        else if (result == 2) folder.revealToUser();
        else if (result == 3) h.rescanVoices();
        else if (result == 4) juce::URL ("https://github.com/lancelotblanchard/rvc-mlx/blob/main/docs/converting-models.md").launchInDefaultBrowser();
    });
}

//==============================================================================================================
void RvcEditor::paint (juce::Graphics& g)
{
    g.fillAll (Palette::background);

    // Header.
    auto header = getLocalBounds().removeFromTop (60).reduced (kPad, 0);
    g.setColour (Palette::text);
    g.setFont (uiFont (21.0f, true));
    const juce::String name = "RVC Morph";
    g.drawText (name, header, juce::Justification::centredLeft);
    const int nameWidth = textWidth (uiFont (21.0f, true), name);
    g.setColour (Palette::muted);
    g.setFont (uiFont (12.0f));
    g.drawText ("MLX", header.withTrimmedLeft (nameWidth + 8), juce::Justification::centredLeft);

    // Status pill.
    if (! statusArea.isEmpty())
    {
        const auto pill = statusArea.toFloat();
        g.setColour (Palette::card);
        g.fillRoundedRectangle (pill, pill.getHeight() / 2.0f);
        if (status.progress >= 0.0f)
        {
            g.setColour (Palette::voiceA.withAlpha (0.18f));
            g.fillRoundedRectangle (pill.withWidth (pill.getWidth() * juce::jlimit (0.0f, 1.0f, status.progress)), pill.getHeight() / 2.0f);
        }
        using Tone = RvcProcessor::Status::Tone;
        const auto dot = status.tone == Tone::ok ? Palette::ok : status.tone == Tone::warning ? Palette::warning
                       : status.tone == Tone::busy ? Palette::voiceA : Palette::muted;
        auto inner = statusArea.reduced (12, 0);
        g.setColour (dot);
        g.fillEllipse (inner.removeFromLeft (8).withSizeKeepingCentre (8, 8).toFloat());
        inner.removeFromLeft (8);
        g.setColour (Palette::text);
        g.setFont (uiFont (12.5f));
        g.drawFittedText (status.text, inner, juce::Justification::centredLeft, 1);
    }

    // Blend labels.
    auto labels = blendArea.withHeight (16);
    g.setFont (uiFont (11.5f, true));
    g.setColour (Palette::voiceA);
    g.drawText ("A", labels, juce::Justification::centredLeft);
    g.setColour (Palette::voiceB);
    g.drawText ("B", labels, juce::Justification::centredRight);
    g.setColour (blend.isEnabled() ? Palette::text : Palette::muted);
    const int pct = juce::roundToInt (blend.getValue() * 100.0);
    g.drawText (blend.isEnabled() ? "MORPH  " + juce::String (100 - pct) + " / " + juce::String (pct) : juce::String ("MORPH"),
                labels, juce::Justification::centred);
    if (blendHint.isNotEmpty())
    {
        g.setColour (Palette::muted);
        g.setFont (uiFont (12.0f));
        g.drawText (blendHint, blendArea.withTop (blendArea.getBottom() - 16), juce::Justification::centred);
    }

    // Footer.
    g.setColour (Palette::muted);
    g.setFont (uiFont (11.5f));
    const juce::String mode = processor.isUsingARA() ? "ARA: clips are converted in the background, then play back instantly"
                                                     : "Live mode: for the best quality, use ARA (Studio One, Cubase, Logic, REAPER)";
    g.drawText (mode, getLocalBounds().removeFromBottom (30).reduced (kPad, 0), juce::Justification::centred);
}

void RvcEditor::resized()
{
    auto r = getLocalBounds().reduced (kPad, 0);
    auto header = r.removeFromTop (60);
    settingsButton.setBounds (header.removeFromRight (80).withSizeKeepingCentre (80, 30));
    header.removeFromRight (10);
    statusArea = header.removeFromRight (290).withSizeKeepingCentre (290, 30);

    r.removeFromTop (6);
    auto cards = r.removeFromTop (112);
    const int cardWidth = (cards.getWidth() - 16) / 2;
    cardA.setBounds (cards.removeFromLeft (cardWidth));
    cardB.setBounds (cards.removeFromRight (cardWidth));

    r.removeFromTop (18);
    blendArea = r.removeFromTop (64);
    blend.setBounds (blendArea.withTrimmedTop (18).withHeight (26).reduced (2, 0));

    r.removeFromTop (6);
    auto knobs = r.removeFromTop (130);
    const int w = knobs.getWidth() / 5;
    for (auto* k : { &pitch, &index, &protect, &mix, &output })
        k->setBounds (knobs.removeFromLeft (w).reduced (10, 0));
}
