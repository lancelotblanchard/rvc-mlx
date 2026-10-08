import AVFoundation
import Foundation

enum AudioIO {
    /// Reads any file AVFoundation understands as mono float samples at its native rate.
    static func readMono(_ url: URL) throws -> (samples: [Float], sampleRate: Double) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        try file.read(into: buffer)
        let frames = Int(buffer.frameLength), channels = Int(format.channelCount)
        var mono = [Float](repeating: 0, count: frames)
        if let data = buffer.floatChannelData {
            for c in 0 ..< channels {
                let ch = data[c]
                for i in 0 ..< frames { mono[i] += ch[i] / Float(channels) }
            }
        }
        return (mono, format.sampleRate)
    }

    /// Writes 16-bit mono WAV (small, and shares cleanly to any app).
    static func writeWav(_ samples: [Float], sampleRate: Double, to url: URL) throws {
        try? FileManager.default.removeItem(at: url)
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: sampleRate,
                                       AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                                       AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw CocoaError(.fileWriteUnknown)
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        try file.write(from: buffer)
    }

    /// Peak envelope for drawing (values 0...1).
    static func envelope(_ samples: [Float], bars: Int) -> [Float] {
        guard !samples.isEmpty, bars > 0 else { return [] }
        let step = max(1, samples.count / bars)
        let peaks = stride(from: 0, to: samples.count, by: step).map { start in
            samples[start ..< min(samples.count, start + step)].reduce(0) { max($0, abs($1)) }
        }
        let top = max(peaks.max() ?? 1, 1e-4)
        return peaks.map { min(1, $0 / top) }
    }
}

/// Microphone recorder with live level metering.
@MainActor
final class Recorder: NSObject, ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var levels: [Float] = []
    @Published private(set) var elapsed: TimeInterval = 0
    @Published var permissionDenied = false

    private var recorder: AVAudioRecorder?
    private var timer: Timer?
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("recording.wav")

    func start() async -> Bool {
        guard await AVAudioApplication.requestRecordPermission() else {
            permissionDenied = true
            return false
        }
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothA2DP])
            try session.setActive(true)
            let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 48000,
                                           AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                                           AVLinearPCMIsFloatKey: false]
            let r = try AVAudioRecorder(url: url, settings: settings)
            r.isMeteringEnabled = true
            r.record()
            recorder = r
            levels = []
            elapsed = 0
            isRecording = true
            timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
                Task { @MainActor in self?.tick() }
            }
            return true
        } catch {
            return false
        }
    }

    private func tick() {
        guard let r = recorder else { return }
        r.updateMeters()
        let db = r.averagePower(forChannel: 0)
        levels.append(max(0, min(1, (db + 50) / 50)))
        if levels.count > 120 { levels.removeFirst(levels.count - 120) }
        elapsed = r.currentTime
    }

    func stop() {
        recorder?.stop()
        recorder = nil
        timer?.invalidate()
        timer = nil
        isRecording = false
    }
}

/// Plays one file at a time and publishes progress.
@MainActor
final class Player: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var playing: URL?
    @Published private(set) var progress: Double = 0
    private var player: AVAudioPlayer?
    private var timer: Timer?

    func toggle(_ url: URL) {
        if playing == url { stop(); return }
        stop()
        try? AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        guard let p = try? AVAudioPlayer(contentsOf: url) else { return }
        p.delegate = self
        p.play()
        player = p
        playing = url
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let p = self.player, p.duration > 0 else { return }
                self.progress = p.currentTime / p.duration
            }
        }
    }

    func stop() {
        player?.stop()
        player = nil
        timer?.invalidate()
        timer = nil
        playing = nil
        progress = 0
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.stop() }
    }
}
