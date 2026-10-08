import Foundation
import MLX

public struct ConvertOptions: Sendable {
    public var pitch: Float = 0          // semitones
    public var indexRate: Float = 0.75   // 0...1
    public var protect: Float = 0.33     // 0...0.5
    public var rmsMixRate: Float = 0.25  // 0...1
    public var speaker = 0
    public var deterministic = false

    public init(pitch: Float = 0, indexRate: Float = 0.75, protect: Float = 0.33, rmsMixRate: Float = 0.25,
                speaker: Int = 0, deterministic: Bool = false) {
        self.pitch = pitch
        self.indexRate = indexRate
        self.protect = protect
        self.rmsMixRate = rmsMixRate
        self.speaker = speaker
        self.deterministic = deterministic
    }
}

/// Chunking geometry in seconds (rvc_mlx.pipeline.PipelineConfig).
public struct ChunkingConfig: Sendable {
    public var xPad = 1, xQuery = 6, xCenter = 38, xMax = 41
    public init(xPad: Int = 1, xQuery: Int = 6, xCenter: Int = 38, xMax: Int = 41) {
        self.xPad = xPad
        self.xQuery = xQuery
        self.xCenter = xCenter
        self.xMax = xMax
    }
}

/// What kind of converted file a URL holds, read from its header only (cheap; for importing).
public enum ModelKind: String, Sendable {
    case hubert, rmvpe, voice

    public static func of(_ url: URL) -> (kind: ModelKind, metadata: [String: String])? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let sizeData = try? handle.read(upToCount: 8), sizeData.count == 8 else { return nil }
        let size = sizeData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard size > 0, size < 100_000_000, let header = try? handle.read(upToCount: Int(size)),
              let json = try? JSONSerialization.jsonObject(with: header) as? [String: Any],
              let meta = json["__metadata__"] as? [String: String],
              let kind = meta["kind"].flatMap(ModelKind.init(rawValue:)) else { return nil }
        return (kind, meta)
    }
}

/// The voice-independent half of RVC (content encoder + pitch estimator) and the conversion pipeline.
/// Not thread-safe: call it from one task / queue at a time.
public final class RVCEngine: @unchecked Sendable {
    public var chunking = ChunkingConfig()
    private let hubert: Hubert
    private let rmvpe: Rmvpe
    private static let sr = 16000, window = 160

    public init(hubertURL: URL, rmvpeURL: URL, precision: Precision = .float16) throws {
        hubert = try Hubert(Weights.load(hubertURL, kind: "hubert", dtype: precision.dtype))
        rmvpe = try Rmvpe(Weights.load(rmvpeURL, kind: "rmvpe", dtype: precision.dtype, keepFloat32: ["mel_basis"]))
    }

    /// RMVPE f0 (Hz, 0 = unvoiced) per 10 ms.
    public func pitch(_ audio16k: [Float]) -> [Float] { rmvpe.f0(audio16k).map { Float($0) } }

    /// The full RVC pipeline over 16 kHz mono audio. Returns audio at `voice.info.sampleRate`.
    /// `progress` receives 0...1 and returns false to cancel.
    public func convert(_ audio16k: [Float], voice: Voice, options: ConvertOptions = .init(),
                        progress: ((Double) -> Bool)? = nil) throws -> [Float] {
        let sr = RVCEngine.sr, window = RVCEngine.window
        guard audio16k.count >= sr / 10 else { throw RVCError.audioTooShort }
        let tgtSr = voice.config.sr
        let tPad = sr * chunking.xPad, tPadTgt = tgtSr * chunking.xPad, tPad2 = 2 * tPad

        let audio = highpass(audio16k.map { Double($0) })
        let optTs = splitPoints(audio, window: window, tMax: sr * chunking.xMax, tCenter: sr * chunking.xCenter,
                                tQuery: sr * chunking.xQuery)
        let pad = reflectPad(audio, left: tPad, right: tPad)
        let pLen = pad.count / window

        var coarse: [Int] = [], hz: [Float] = []
        if voice.config.f0 {
            (coarse, hz) = pitchTrack(pad, semitones: options.pitch)
            coarse = Array(coarse.prefix(pLen))
            hz = Array(hz.prefix(pLen))
        }

        var out: [Float] = []
        let expected = max(1.0, Double(audio.count) * Double(tgtSr) / Double(sr))
        func runChunk(_ s: Int, _ e: Int, pitchStart: Int) throws {
            let ps = min(pitchStart, coarse.count)
            let chunk = try vc(voice, options, Array(pad[s ..< e]),
                               pitch: Array(coarse[ps...]), pitchf: Array(hz[ps...]))
            if chunk.count > 2 * tPadTgt { out.append(contentsOf: chunk[tPadTgt ..< (chunk.count - tPadTgt)]) }
            if let progress, !progress(min(1, Double(out.count) / expected)) { throw RVCError.cancelled }
        }
        var s = 0
        var t: Int? = nil
        for ts in optTs {
            let tt = ts / window * window
            try runChunk(s, min(pad.count, tt + tPad2 + window), pitchStart: s / window)
            s = tt
            t = tt
        }
        try runChunk(t ?? 0, pad.count, pitchStart: (t ?? 0) / window)

        if options.rmsMixRate != 1 {
            changeRms(source: audio, sourceRate: sr, target: &out, targetRate: tgtSr, rate: Double(options.rmsMixRate))
        }
        let peak = out.reduce(0) { max($0, abs($1)) }
        if peak / 0.99 > 1 { out = out.map { $0 / (peak / 0.99) } }
        _ = progress?(1)
        return out
    }

    private func pitchTrack(_ audio: [Double], semitones: Float) -> ([Int], [Float]) {
        let factor = pow(2.0, Double(semitones) / 12)
        let f0 = rmvpe.f0(audio.map { Float($0) }).map { $0 * factor }
        return (coarseF0(f0), f0.map { Float($0) })
    }

    /// One RVC `vc` call: features -> retrieval -> protect -> synthesis.
    private func vc(_ voice: Voice, _ opt: ConvertOptions, _ audio: [Double], pitch: [Int], pitchf: [Float]) throws -> [Float] {
        let hasPitch = voice.config.f0
        let x = MLXArray(audio.map { Float($0) }, [1, audio.count])
        var feats = try hubert.features(x, version: voice.config.version).asType(.float32)
        var feats0 = feats
        if !voice.banks.isEmpty && opt.indexRate != 0 {
            feats = retrieveBlended(feats, banks: voice.banks, weights: voice.bankWeights) * opt.indexRate
                + feats * (1 - opt.indexRate)
        }
        feats = repeated(feats, count: 2, axis: 1)
        feats0 = repeated(feats0, count: 2, axis: 1)

        var pLen = min(audio.count / RVCEngine.window, feats.dim(1))
        if hasPitch { pLen = min(pLen, pitch.count) }
        feats = timeSlice(feats, 0, pLen)

        let spk = min(max(opt.speaker, 0), voice.config.spkEmbedDim - 1)
        var inputs = SynthInputs(phone: feats, length: pLen, speaker: voice.weights["emb_g.weight"][spk ..< (spk + 1)])
        if hasPitch {
            inputs.pitch = int32Array(Array(pitch.prefix(pLen)), [1, pLen])
            inputs.pitchf = MLXArray(Array(pitchf.prefix(pLen)), [1, pLen])
            if opt.protect < 0.5 {
                let keep = MLXArray(pitchf.prefix(pLen).map { $0 < 1 ? opt.protect : 1 }, [1, pLen, 1])
                feats = feats * keep + timeSlice(feats0, 0, pLen) * (1 - keep)
            }
        }
        inputs.phone = feats.asType(voice.weights["enc_p.emb_phone.weight"].dtype)
        if opt.deterministic {
            inputs.noiseScale = 0
            inputs.nsfNoiseScale = 0
        }
        let y = synthesize(voice.weights, voice.config, inputs).asType(.float32)
        eval(y)
        return y.asArray(Float.self)
    }
}
