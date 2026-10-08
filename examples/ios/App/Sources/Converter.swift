import Foundation
import RVCKit

/// Runs conversions off the main thread and publishes their state.
@MainActor
final class Converter: ObservableObject {
    enum Phase: Equatable {
        case idle
        case loading                  // loading models into memory (first run)
        case converting(Double)       // 0...1
        case done
        case failed(String)
    }

    struct Result: Equatable {
        let url: URL
        let envelope: [Float]
        let duration: Double
        let seconds: Double           // processing time
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var result: Result?

    private let worker = Worker()
    private var task: Task<Void, Never>?

    var isBusy: Bool {
        if case .converting = phase { return true }
        return phase == .loading
    }

    func convert(input: URL, voice: URL, hubert: URL, rmvpe: URL, options: ConvertOptions) {
        task?.cancel()
        result = nil
        phase = .loading
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(voice.deletingPathExtension().lastPathComponent)-\(Int(Date().timeIntervalSince1970)).wav")
        task = Task {
            do {
                let r = try await worker.run(input: input, voice: voice, hubert: hubert, rmvpe: rmvpe, options: options,
                                             output: output) { [weak self] p in
                    Task { @MainActor in
                        guard let self else { return }
                        if case .converting(let old) = self.phase, p - old < 0.01, p < 1 { return }
                        self.phase = .converting(p)
                    }
                }
                result = r
                phase = .done
            } catch is CancellationError {
                phase = .idle
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        task?.cancel()
    }
}

/// Owns the MLX engine and voice cache; serialises all model work.
private actor Worker {
    private var engine: RVCEngine?
    private var engineKey: [URL] = []
    private var voices: [URL: Voice] = [:]

    func run(input: URL, voice voiceURL: URL, hubert: URL, rmvpe: URL, options: ConvertOptions, output: URL,
             progress: @escaping @Sendable (Double) -> Void) throws -> Converter.Result {
        let start = Date()
        if engine == nil || engineKey != [hubert, rmvpe] {
            engine = try RVCEngine(hubertURL: hubert, rmvpeURL: rmvpe, precision: .float16)
            engineKey = [hubert, rmvpe]
        }
        if voices[voiceURL] == nil {
            if voices.count >= 2 { voices.removeAll() }  // keep memory bounded on phones
            voices[voiceURL] = try Voice.load(voiceURL, precision: .float16)
        }
        guard let engine, let voice = voices[voiceURL] else { throw CancellationError() }
        progress(0)

        let (samples, rate) = try AudioIO.readMono(input)
        let audio16k = resample(samples, from: rate, to: 16000)
        let converted = try engine.convert(audio16k, voice: voice, options: options) { p in
            progress(p)
            return !Task.isCancelled
        }
        try Task.checkCancellation()
        let sr = Double(voice.info.sampleRate)
        try AudioIO.writeWav(converted, sampleRate: sr, to: output)
        return .init(url: output, envelope: AudioIO.envelope(converted, bars: 64),
                     duration: Double(converted.count) / sr, seconds: Date().timeIntervalSince(start))
    }
}
