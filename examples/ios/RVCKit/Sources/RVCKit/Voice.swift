import Foundation
import MLX

public struct VoiceInfo: Sendable, Hashable {
    public var name: String
    public var version: String     // "v1" or "v2"
    public var sampleRate: Int
    public var hasPitch: Bool
    public var indexSize: Int
    public var mergedFrom: String
}

/// A converted RVC voice (synthesizer weights + retrieval banks).
public final class Voice: @unchecked Sendable {
    public internal(set) var info: VoiceInfo
    var config: SynthConfig
    var weights: Weights
    var banks: [MLXArray]        // (N, featureDim) each, float32
    var bankWeights: [Float]

    init(info: VoiceInfo, config: SynthConfig, weights: Weights, banks: [MLXArray], bankWeights: [Float]) {
        self.info = info
        self.config = config
        self.weights = weights
        self.banks = banks
        self.bankWeights = bankWeights
    }

    public static func load(_ url: URL, precision: Precision = .float16) throws -> Voice {
        var w = try Weights.load(url, kind: "voice", dtype: precision.dtype, keepFloat32: ["index.vectors"])
        try w.require(["enc_p.emb_phone.weight", "dec.conv_pre.weight", "emb_g.weight", "flow.flows.0.pre.weight"])
        let config = try SynthConfig(json: w.json("config"))
        var banks: [MLXArray] = [], bankWeights: [Float] = []
        if let index = w.arrays.removeValue(forKey: "index.vectors") {
            banks = [index]
            bankWeights = [1]
        }
        let info = VoiceInfo(
            name: w.metadata["name"] ?? url.deletingPathExtension().lastPathComponent,
            version: config.version, sampleRate: config.sr, hasPitch: config.f0,
            indexSize: banks.first?.dim(0) ?? 0, mergedFrom: w.metadata["merged_from"] ?? "")
        return Voice(info: info, config: config, weights: w, banks: banks, bankWeights: bankWeights)
    }

    /// Why two voices can't be blended (nil when they can).
    public static func blendProblem(_ a: Voice, _ b: Voice) -> String? {
        if a.config.version != b.config.version { return "different RVC versions (\(a.config.version) vs \(b.config.version))" }
        if a.config.sr != b.config.sr { return "different sample rates (\(a.config.sr / 1000)k vs \(b.config.sr / 1000)k)" }
        if a.config.f0 != b.config.f0 { return "one voice uses pitch guidance and the other doesn't" }
        if !a.config.sameArchitecture(b.config) { return "different model sizes" }
        return nil
    }

    /// Weight interpolation (RVC "ckpt merge"); retrieval banks are kept separately and mixed by the same weights.
    public static func blend(_ voices: [Voice], weights: [Float]) throws -> Voice {
        guard let ref = voices.first, voices.count == weights.count else { throw RVCError.incompatibleVoices("need one weight per voice") }
        for v in voices.dropFirst() {
            if let why = blendProblem(ref, v) { throw RVCError.incompatibleVoices(why) }
        }
        let total = weights.reduce(0, +)
        guard total > 0 else { throw RVCError.incompatibleVoices("weights must sum to a positive number") }
        let ws = weights.map { $0 / total }

        var arrays: [String: MLXArray] = [:]
        for (key, first) in ref.weights.arrays {
            let rows = key == "emb_g.weight" ? voices.map { $0.weights[key].dim(0) }.min() : nil
            var acc: MLXArray? = nil
            for (v, wt) in zip(voices, ws) where wt != 0 {
                var a = v.weights[key].asType(.float32)
                if let rows { a = a[0 ..< rows] }
                acc = acc.map { $0 + a * wt } ?? a * wt
            }
            arrays[key] = acc!.asType(first.dtype)
        }
        eval(Array(arrays.values))

        var config = ref.config
        config.spkEmbedDim = arrays["emb_g.weight"]?.dim(0) ?? config.spkEmbedDim
        var banks: [MLXArray] = [], bankWeights: [Float] = []
        for (v, wt) in zip(voices, ws) where wt != 0 {
            for (b, bw) in zip(v.banks, v.bankWeights) {
                banks.append(b)
                bankWeights.append(wt * bw)
            }
        }
        let recipe = zip(voices, ws).map { "\($0.info.name):\(String(format: "%.3f", $1))" }.joined(separator: ", ")
        var info = ref.info
        info.name = voices.map(\.info.name).joined(separator: " + ")
        info.mergedFrom = recipe
        info.indexSize = banks.reduce(0) { $0 + $1.dim(0) }
        let w = Weights(arrays: arrays, metadata: ref.weights.metadata, file: "<blend>")
        return Voice(info: info, config: config, weights: w, banks: banks, bankWeights: bankWeights)
    }
}

// MARK: - Retrieval

private func retrieve(_ q: MLXArray, _ bank: MLXArray, k kIn: Int) -> MLXArray {    // q (T, D), bank (N, D)
    let k = min(kIn, bank.dim(0))
    let bankSq = (bank * bank).sum(axis: -1)
    var outs: [MLXArray] = []
    let chunk = 512
    for s in stride(from: 0, to: q.dim(0), by: chunk) {
        let qc = q[s ..< min(q.dim(0), s + chunk)]
        let d = (qc * qc).sum(axis: -1, keepDims: true) - 2 * matmul(qc, bank.T) + bankSq.expandedDimensions(axis: 0)
        let idx = argPartition(d, kth: k - 1, axis: -1)[0..., 0 ..< k]                  // (t, k)
        let dist = maximum(takeAlong(d, idx, axis: -1), Float(0))
        var w = 1 / square(maximum(dist, Float(1e-12)))
        w = w / w.sum(axis: -1, keepDims: true)
        let rows = take(bank, idx.reshaped(-1), axis: 0).reshaped(idx.dim(0), k, bank.dim(1))
        outs.append((rows * w.expandedDimensions(axis: -1)).sum(axis: 1))
    }
    return concatenated(outs, axis: 0)
}

/// RVC's index retrieval against several banks: sum_i w_i * retrieve(feats, bank_i). feats: (1, T, D).
func retrieveBlended(_ feats: MLXArray, banks: [MLXArray], weights: [Float], k: Int = 8) -> MLXArray {
    let q = feats[0].asType(.float32)
    let total = weights.reduce(0, +)
    var acc: MLXArray? = nil
    for (bank, wt) in zip(banks, weights) where wt != 0 {
        let r = retrieve(q, bank, k: k) * (wt / total)
        acc = acc.map { $0 + r } ?? r
    }
    return acc!.asType(feats.dtype).expandedDimensions(axis: 0)
}
