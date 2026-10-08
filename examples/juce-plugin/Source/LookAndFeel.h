#pragma once

#include <JuceHeader.h>

namespace Palette
{
inline const juce::Colour background { 0xff111114 };
inline const juce::Colour card { 0xff1b1b20 };
inline const juce::Colour cardEdge { 0xff2a2a31 };
inline const juce::Colour text { 0xfff4f4f5 };
inline const juce::Colour muted { 0xff8b8b95 };
inline const juce::Colour track { 0xff2c2c33 };
inline const juce::Colour voiceA { 0xffff5a36 };  // matches the web UI accent
inline const juce::Colour voiceB { 0xff8b7cff };
inline const juce::Colour ok { 0xff3ecf8e };
inline const juce::Colour warning { 0xfff5a524 };
} // namespace Palette

inline juce::Font uiFont (float size, bool bold = false)
{
   #if JUCE_MAJOR_VERSION >= 8
    return juce::Font (juce::FontOptions (size, bold ? juce::Font::bold : juce::Font::plain));
   #else
    return juce::Font (size, bold ? juce::Font::bold : juce::Font::plain);
   #endif
}

inline int textWidth (const juce::Font& font, const juce::String& text)
{
   #if JUCE_MAJOR_VERSION >= 8
    return juce::GlyphArrangement::getStringWidthInt (font, text);
   #else
    return font.getStringWidth (text);
   #endif
}

class RvcLookAndFeel : public juce::LookAndFeel_V4
{
public:
    RvcLookAndFeel()
    {
        setColour (juce::ComboBox::backgroundColourId, Palette::track);
        setColour (juce::ComboBox::textColourId, Palette::text);
        setColour (juce::ComboBox::outlineColourId, juce::Colours::transparentBlack);
        setColour (juce::ComboBox::arrowColourId, Palette::muted);
        setColour (juce::PopupMenu::backgroundColourId, Palette::card);
        setColour (juce::PopupMenu::textColourId, Palette::text);
        setColour (juce::PopupMenu::highlightedBackgroundColourId, Palette::track);
        setColour (juce::PopupMenu::highlightedTextColourId, Palette::text);
        setColour (juce::Slider::textBoxTextColourId, Palette::text);
        setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        setColour (juce::Label::textColourId, Palette::text);
        setColour (juce::TextButton::buttonColourId, Palette::track);
        setColour (juce::TextButton::textColourOffId, Palette::text);
    }

    void drawRotarySlider (juce::Graphics& g, int x, int y, int w, int h, float pos, float start, float end, juce::Slider& s) override
    {
        const auto bounds = juce::Rectangle<int> (x, y, w, h).toFloat().reduced (6.0f);
        const float radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) / 2.0f;
        const auto centre = bounds.getCentre();
        const float thickness = 3.5f;
        const auto accent = s.findColour (juce::Slider::rotarySliderFillColourId);

        juce::Path track;
        track.addCentredArc (centre.x, centre.y, radius - thickness, radius - thickness, 0.0f, start, end, true);
        g.setColour (Palette::track);
        g.strokePath (track, juce::PathStrokeType (thickness, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        // Bipolar parameters (pitch) fill from the centre; the rest from the start.
        const bool bipolar = s.getMinimum() < 0.0 && s.getMaximum() > 0.0;
        const float from = bipolar ? (start + end) / 2.0f : start;
        const float to = start + pos * (end - start);
        if (std::abs (to - from) > 0.001f)
        {
            juce::Path value;
            value.addCentredArc (centre.x, centre.y, radius - thickness, radius - thickness, 0.0f, juce::jmin (from, to), juce::jmax (from, to), true);
            g.setColour (s.isEnabled() ? accent : Palette::muted);
            g.strokePath (value, juce::PathStrokeType (thickness, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        }

        const float knob = radius - thickness * 3.0f;
        g.setColour (Palette::card.brighter (0.08f));
        g.fillEllipse (juce::Rectangle<float> (knob * 2, knob * 2).withCentre (centre));
        const auto tip = centre.getPointOnCircumference (knob - 5.0f, to);
        g.setColour (Palette::text);
        g.drawLine ({ centre.getPointOnCircumference (knob * 0.35f, to), tip }, 2.5f);
    }

    void drawLinearSlider (juce::Graphics& g, int x, int y, int w, int h, float sliderPos, float, float,
                           juce::Slider::SliderStyle, juce::Slider& s) override
    {
        const auto area = juce::Rectangle<int> (x, y, w, h).toFloat();
        const auto trackArea = area.withSizeKeepingCentre (area.getWidth(), 6.0f);
        const bool enabled = s.isEnabled();
        g.setColour (Palette::track);
        g.fillRoundedRectangle (trackArea, 3.0f);
        if (enabled)
        {
            juce::ColourGradient grad (Palette::voiceA, trackArea.getX(), 0, Palette::voiceB, trackArea.getRight(), 0, false);
            g.setGradientFill (grad);
            g.setOpacity (0.85f);
            g.fillRoundedRectangle (trackArea, 3.0f);
            g.setOpacity (1.0f);
        }
        const float thumb = 18.0f;
        const auto thumbArea = juce::Rectangle<float> (thumb, thumb).withCentre ({ sliderPos, area.getCentreY() });
        const auto mixColour = Palette::voiceA.interpolatedWith (Palette::voiceB, (float) s.valueToProportionOfLength (s.getValue()));
        g.setColour (enabled ? juce::Colours::white : Palette::muted);
        g.fillEllipse (thumbArea);
        g.setColour (enabled ? mixColour : Palette::track);
        g.fillEllipse (thumbArea.reduced (5.0f));
    }

    void drawComboBox (juce::Graphics& g, int w, int h, bool, int, int, int, int, juce::ComboBox& box) override
    {
        const auto r = juce::Rectangle<int> (w, h).toFloat();
        g.setColour (box.findColour (juce::ComboBox::backgroundColourId));
        g.fillRoundedRectangle (r, 9.0f);
        juce::Path arrow;
        const float ax = r.getRight() - 20.0f, ay = r.getCentreY();
        arrow.startNewSubPath (ax - 4.0f, ay - 2.0f);
        arrow.lineTo (ax, ay + 2.5f);
        arrow.lineTo (ax + 4.0f, ay - 2.0f);
        g.setColour (Palette::muted);
        g.strokePath (arrow, juce::PathStrokeType (1.6f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }

    juce::Font getComboBoxFont (juce::ComboBox&) override { return uiFont (15.0f, true); }
    juce::Font getPopupMenuFont() override { return uiFont (14.0f); }

    void positionComboBoxText (juce::ComboBox& box, juce::Label& label) override
    {
        label.setBounds (8, 1, box.getWidth() - 36, box.getHeight() - 2);
        label.setFont (getComboBoxFont (box));
    }

    void drawPopupMenuBackground (juce::Graphics& g, int w, int h) override
    {
        g.fillAll (Palette::card);
        g.setColour (Palette::cardEdge);
        g.drawRect (0, 0, w, h);
    }
};
