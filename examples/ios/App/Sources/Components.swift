import SwiftUI

struct PitchPreset: Identifiable {
    let label: String
    let semitones: Double
    var id: Double { semitones }
    static let all = [PitchPreset(label: "Deeper", semitones: -12), PitchPreset(label: "Same", semitones: 0),
                      PitchPreset(label: "Higher", semitones: 12)]
}

struct Card<Content: View>: View {
    let title: String
    let systemImage: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(title, systemImage: systemImage)
                .font(.footnote.weight(.semibold))
                .textCase(.uppercase)
                .foregroundStyle(.secondary)
            content
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }
}

/// Bars for a level history (live) or a file envelope, with an optional playhead.
struct Waveform: View {
    let levels: [Float]
    let progress: Double?
    let color: Color
    var live = false

    var body: some View {
        GeometryReader { geo in
            let count = live ? 60 : max(levels.count, 1)
            let values = live ? Array(repeating: Float(0), count: max(0, count - levels.count)) + levels.suffix(count) : levels
            let barWidth = max(2, geo.size.width / CGFloat(count) * 0.6)
            HStack(alignment: .center, spacing: geo.size.width / CGFloat(count) - barWidth) {
                ForEach(Array(values.enumerated()), id: \.offset) { i, v in
                    let played = progress.map { Double(i) / Double(values.count) <= $0 } ?? false
                    Capsule()
                        .fill(played ? Color.accentColor : color.opacity(progress == nil ? 0.8 : 0.35))
                        .frame(width: barWidth, height: max(3, CGFloat(v) * geo.size.height))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .animation(.linear(duration: 0.05), value: values)
        }
    }
}

struct RecordButton: View {
    let isRecording: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().stroke(Color.accentColor.opacity(0.35), lineWidth: 4).frame(width: 76, height: 76)
                RoundedRectangle(cornerRadius: isRecording ? 8 : 30, style: .continuous)
                    .fill(Color.accentColor)
                    .frame(width: isRecording ? 30 : 60, height: isRecording ? 30 : 60)
            }
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isRecording)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(isRecording ? "Stop recording" : "Record")
        .sensoryFeedback(.impact, trigger: isRecording)
    }
}

struct CircleButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .frame(width: 48, height: 48)
            .background(Color(.tertiarySystemFill), in: Circle())
            .foregroundStyle(isEnabled ? Color.primary : Color.secondary.opacity(0.5))
            .scaleEffect(configuration.isPressed ? 0.94 : 1)
    }
}

struct VoiceChip: View {
    let item: ModelLibrary.VoiceItem
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 2) {
                Text(item.name).font(.subheadline.weight(.semibold)).lineLimit(1)
                Text(item.detail).font(.caption2).opacity(0.75).lineLimit(1)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .foregroundStyle(selected ? Color.white : Color.primary)
            .background(selected ? Color.accentColor : Color(.tertiarySystemFill),
                        in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
        .animation(.easeOut(duration: 0.15), value: selected)
    }
}

struct LabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let hint: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: "%.2f", value)).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: $value, in: range)
            Text(hint).font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// Shown until the base models and at least one voice are installed.
struct SetupCard: View {
    @EnvironmentObject private var library: ModelLibrary
    let onImport: () -> Void

    var body: some View {
        Card(title: "Set up", systemImage: "shippingbox") {
            VStack(alignment: .leading, spacing: 12) {
                Text("RVC Pocket runs your RVC voices on this iPhone. Convert them on a Mac with rvc-mlx, then send the files here with AirDrop, the Files app, or Import.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                step("Content encoder", "hubert.safetensors", done: library.hubertURL != nil)
                step("Pitch model", "rmvpe.safetensors", done: library.rmvpeURL != nil)
                step("A voice", library.voices.isEmpty ? "voices/<name>.safetensors" : "\(library.voices.count) installed",
                     done: !library.voices.isEmpty)
                Button(action: onImport) {
                    Label("Import models", systemImage: "square.and.arrow.down").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }
        }
    }

    private func step(_ title: String, _ detail: String, done: Bool) -> some View {
        HStack(spacing: 12) {
            Image(systemName: done ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(done ? Color.green : Color.secondary)
                .font(.title3)
            VStack(alignment: .leading, spacing: 0) {
                Text(title).font(.subheadline.weight(.medium))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
