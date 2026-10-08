import RVCKit
import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    static let safetensors = UTType(importedAs: "com.rvc-mlx.safetensors", conformingTo: .data)
}

struct ContentView: View {
    @EnvironmentObject private var library: ModelLibrary
    @EnvironmentObject private var converter: Converter
    @StateObject private var recorder = Recorder()
    @StateObject private var player = Player()

    @State private var input: URL?
    @State private var inputEnvelope: [Float] = []
    @State private var inputDuration: Double = 0
    @State private var voice: URL?
    @State private var pitch: Double = 0
    @State private var indexRate: Double = 0.75
    @State private var protect: Double = 0.33
    @State private var showAdvanced = false
    @State private var importing: ImportMode?          // drives presentation (cleared on dismiss)
    @State private var importKind: ImportMode = .models  // what the open picker is for (survives dismissal)

    private enum ImportMode { case models, audio }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if !library.isReady {
                        SetupCard { importKind = .models; importing = .models }
                    }
                    inputCard
                    voiceCard
                    convertButton
                    if let result = converter.result {
                        resultCard(result)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 32)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("RVC Pocket")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button("Import models…", systemImage: "square.and.arrow.down") { importKind = .models; importing = .models }
                        Button("Refresh library", systemImage: "arrow.clockwise") { library.refresh() }
                        Link(destination: URL(string: "https://github.com/lancelotblanchard/rvc-mlx/blob/main/docs/converting-models.md")!) {
                            Label("How to convert voices", systemImage: "questionmark.circle")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            // One importer for both uses: SwiftUI only honours a single .fileImporter per view.
            .fileImporter(isPresented: Binding(get: { importing != nil }, set: { if !$0 { importing = nil } }),
                          allowedContentTypes: importKind == .audio ? [.audio] : [.safetensors, .data],
                          allowsMultipleSelection: importKind == .models) { result in
                guard case .success(let urls) = result else { return }
                if importKind == .audio, let url = urls.first {
                    setInput(copying: url)
                } else {
                    library.importFiles(urls)
                }
            }
            .onChange(of: importing) { _, mode in if let mode { importKind = mode } }
            .alert("Microphone access", isPresented: $recorder.permissionDenied) {
                Button("OK", role: .cancel) {}
            } message: {
                Text("Allow microphone access in Settings to record.")
            }
            .alert(library.importMessage ?? "", isPresented: Binding(get: { library.importMessage != nil },
                                                                    set: { if !$0 { library.importMessage = nil } })) {
                Button("OK", role: .cancel) {}
            }
            .onAppear { if voice == nil { voice = library.voices.first?.url } }
            .onChange(of: library.voices) { _, voices in
                if voice == nil || !voices.contains(where: { $0.url == voice }) { voice = voices.first?.url }
            }
        }
    }

    // MARK: - Input

    private var inputCard: some View {
        Card(title: "Your voice", systemImage: "waveform") {
            VStack(spacing: 14) {
                ZStack {
                    if recorder.isRecording {
                        Waveform(levels: recorder.levels, progress: nil, color: .accentColor, live: true)
                    } else if !inputEnvelope.isEmpty {
                        Waveform(levels: inputEnvelope, progress: player.playing == input ? player.progress : nil, color: .secondary)
                    } else {
                        Text("Record or import something to convert")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(height: 64)

                HStack(spacing: 20) {
                    Button { importKind = .audio; importing = .audio } label: {
                        Image(systemName: "folder").font(.title3)
                    }
                    .buttonStyle(CircleButtonStyle())
                    .disabled(recorder.isRecording)

                    RecordButton(isRecording: recorder.isRecording) {
                        Task { await toggleRecording() }
                    }

                    Button {
                        if let input { player.toggle(input) }
                    } label: {
                        Image(systemName: player.playing == input && input != nil ? "stop.fill" : "play.fill").font(.title3)
                    }
                    .buttonStyle(CircleButtonStyle())
                    .disabled(input == nil || recorder.isRecording)
                }
                Text(recorder.isRecording ? timeString(recorder.elapsed) : (input == nil ? " " : timeString(inputDuration)))
                    .font(.footnote.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func toggleRecording() async {
        if recorder.isRecording {
            recorder.stop()
            setInput(recorder.url)
        } else {
            player.stop()
            _ = await recorder.start()
        }
    }

    private func setInput(copying picked: URL) {
        let scoped = picked.startAccessingSecurityScopedResource()
        defer { if scoped { picked.stopAccessingSecurityScopedResource() } }
        let local = FileManager.default.temporaryDirectory.appendingPathComponent("input-" + picked.lastPathComponent)
        try? FileManager.default.removeItem(at: local)
        guard (try? FileManager.default.copyItem(at: picked, to: local)) != nil else { return }
        setInput(local)
    }

    private func setInput(_ url: URL) {
        guard let decoded = try? AudioIO.readMono(url) else { return }
        input = url
        inputEnvelope = AudioIO.envelope(decoded.samples, bars: 64)
        inputDuration = Double(decoded.samples.count) / decoded.sampleRate
    }

    // MARK: - Voice

    private var voiceCard: some View {
        Card(title: "Convert to", systemImage: "person.wave.2") {
            VStack(alignment: .leading, spacing: 16) {
                if library.voices.isEmpty {
                    Text("No voices yet. Import a converted voice (.safetensors).")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                } else {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 10) {
                            ForEach(library.voices) { item in
                                VoiceChip(item: item, selected: item.url == voice) { voice = item.url }
                                    .contextMenu {
                                        Button("Delete", systemImage: "trash", role: .destructive) { library.delete(item) }
                                    }
                            }
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Pitch")
                        Spacer()
                        Text(pitch == 0 ? "Original" : String(format: "%+.0f semitones", pitch))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    .font(.subheadline)
                    Slider(value: $pitch, in: -12 ... 12, step: 1)
                    HStack {
                        ForEach(PitchPreset.all) { preset in
                            Button(preset.label) { withAnimation { pitch = preset.semitones } }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                                .tint(pitch == preset.semitones ? .accentColor : .secondary)
                        }
                    }
                }

                DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                    VStack(spacing: 10) {
                        LabeledSlider(title: "Index", value: $indexRate, range: 0 ... 1,
                                      hint: "How strongly to pull the timbre toward the voice's training data")
                        LabeledSlider(title: "Protect", value: $protect, range: 0 ... 0.5,
                                      hint: "Keeps consonants and breaths natural (0.5 = off)")
                    }
                    .padding(.top, 8)
                }
                .font(.subheadline)
            }
        }
    }

    // MARK: - Convert

    private var convertButton: some View {
        Button {
            guard let input, let voice, let hubert = library.hubertURL, let rmvpe = library.rmvpeURL else { return }
            player.stop()
            converter.convert(input: input, voice: voice, hubert: hubert, rmvpe: rmvpe,
                              options: ConvertOptions(pitch: Float(pitch), indexRate: Float(indexRate), protect: Float(protect)))
        } label: {
            HStack(spacing: 10) {
                switch converter.phase {
                case .loading:
                    ProgressView().tint(.white)
                    Text("Loading models…")
                case .converting(let p):
                    ProgressView(value: p).progressViewStyle(.circular).tint(.white)
                    Text("Converting \(Int(p * 100)) %")
                default:
                    Image(systemName: "wand.and.stars")
                    Text("Convert")
                }
            }
            .font(.headline)
            .frame(maxWidth: .infinity, minHeight: 54)
        }
        .buttonStyle(.borderedProminent)
        .buttonBorderShape(.roundedRectangle(radius: 16))
        .disabled(input == nil || voice == nil || !library.isReady || converter.isBusy || recorder.isRecording)
        .overlay(alignment: .bottom) {
            if case .failed(let message) = converter.phase {
                Text(message).font(.footnote).foregroundStyle(.red).offset(y: 24)
            }
        }
    }

    private func resultCard(_ result: Converter.Result) -> some View {
        Card(title: "Result", systemImage: "sparkles") {
            VStack(spacing: 12) {
                Waveform(levels: result.envelope, progress: player.playing == result.url ? player.progress : nil, color: .accentColor)
                    .frame(height: 64)
                HStack {
                    Button {
                        player.toggle(result.url)
                    } label: {
                        Label(player.playing == result.url ? "Stop" : "Play",
                              systemImage: player.playing == result.url ? "stop.fill" : "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    ShareLink(item: result.url) {
                        Label("Share", systemImage: "square.and.arrow.up").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
                Text(String(format: "%.1f s converted in %.1f s", result.duration, result.seconds))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .onAppear { player.toggle(result.url) }
    }

    private func timeString(_ t: TimeInterval) -> String { String(format: "%d:%04.1f", Int(t) / 60, t.truncatingRemainder(dividingBy: 60)) }
}
