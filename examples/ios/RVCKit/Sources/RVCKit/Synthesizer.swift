import Foundation
import MLX

/// RVC synthesizer architecture (the `config` JSON stored in converted voices).
struct SynthConfig: Equatable {
    var interChannels = 192, hiddenChannels = 192, filterChannels = 768, nHeads = 2, nLayers = 6, kernelSize = 3
    var resblock = "1"
    var resblockKernelSizes: [Int] = [3, 7, 11]
    var resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    var upsampleRates: [Int] = [10, 10, 2, 2]
    var upsampleInitialChannel = 512
    var upsampleKernelSizes: [Int] = [16, 16, 4, 4]
    var spkEmbedDim = 109, ginChannels = 256, sr = 40000
    var version = "v2"
    var f0 = true

    init(json j: [String: Any]) throws {
        func int(_ k: String) throws -> Int {
            guard let v = j[k] as? Int else { throw RVCError.badConfig("missing \(k)") }
            return v
        }
        interChannels = try int("inter_channels")
        hiddenChannels = try int("hidden_channels")
        filterChannels = try int("filter_channels")
        nHeads = try int("n_heads")
        nLayers = try int("n_layers")
        kernelSize = try int("kernel_size")
        resblock = j["resblock"] as? String ?? "1"
        resblockKernelSizes = j["resblock_kernel_sizes"] as? [Int] ?? resblockKernelSizes
        resblockDilationSizes = j["resblock_dilation_sizes"] as? [[Int]] ?? resblockDilationSizes
        upsampleRates = j["upsample_rates"] as? [Int] ?? upsampleRates
        upsampleInitialChannel = try int("upsample_initial_channel")
        upsampleKernelSizes = j["upsample_kernel_sizes"] as? [Int] ?? upsampleKernelSizes
        spkEmbedDim = try int("spk_embed_dim")
        ginChannels = try int("gin_channels")
        sr = try int("sr")
        version = j["version"] as? String ?? "v2"
        f0 = j["f0"] as? Bool ?? true
    }

    var featureDim: Int { version == "v1" ? 256 : 768 }
    var hopLength: Int { upsampleRates.reduce(1, *) }

    /// Everything that must match for two voices to be blended.
    func sameArchitecture(_ o: SynthConfig) -> Bool {
        var a = self, b = o
        a.spkEmbedDim = 0
        b.spkEmbedDim = 0
        return a == b
    }
}

struct SynthInputs {
    var phone: MLXArray            // (1, T, featureDim)
    var length: Int
    var pitch: MLXArray?           // (1, T) int32 coarse bins
    var pitchf: MLXArray?          // (1, T) float32 Hz
    var speaker: MLXArray          // (1, ginChannels)
    var noiseScale: Float = 0.66666
    var nsfNoiseScale: Float = 1
    var priorNoise: MLXArray? = nil
    var nsfNoise: MLXArray? = nil
}

private let lreluSlope: Float = 0.1
private let windowSize = 10

// MARK: text encoder

private func relativeEmbeddings(_ emb: MLXArray, _ length: Int) -> MLXArray {
    let pad = max(length - (windowSize + 1), 0)
    let start = max((windowSize + 1) - length, 0)
    let e = pad > 0 ? padAxis(emb, axis: 1, before: pad, after: pad) : emb
    return e[0..., start ..< (start + 2 * length - 1)]                       // (1, 2L-1, K)
}

private func relativeToAbsolute(_ x: MLXArray) -> MLXArray {                // (B, H, L, 2L-1) -> (B, H, L, L)
    let B = x.dim(0), H = x.dim(1), L = x.dim(2)
    var y = padAxis(x, axis: 3, before: 0, after: 1).reshaped(B, H, 2 * L * L)
    y = padAxis(y, axis: 2, before: 0, after: L - 1).reshaped(B, H, L + 1, 2 * L - 1)
    return y[0..., 0..., 0 ..< L, (L - 1)...]
}

private func absoluteToRelative(_ x: MLXArray) -> MLXArray {                // (B, H, L, L) -> (B, H, L, 2L-1)
    let B = x.dim(0), H = x.dim(1), L = x.dim(2)
    var y = padAxis(x, axis: 3, before: 0, after: L - 1).reshaped(B, H, L * L + L * (L - 1))
    y = padAxis(y, axis: 2, before: L, after: 0).reshaped(B, H, L, 2 * L)
    return y[.ellipsis, 1...]
}

private func attention(_ w: Weights, _ p: String, _ x: MLXArray, _ mask: MLXArray, _ nHeads: Int) -> MLXArray {
    let B = x.dim(0), T = x.dim(1), C = x.dim(2), K = C / nHeads
    func heads(_ t: MLXArray) -> MLXArray { t.reshaped(B, T, nHeads, K).transposed(0, 2, 1, 3) }
    let q = heads(conv1d(w, p + ".conv_q", x)) * (Float(1) / Float(K).squareRoot())
    let k = heads(conv1d(w, p + ".conv_k", x))
    let v = heads(conv1d(w, p + ".conv_v", x))
    var scores = matmul(q, k.transposed(0, 1, 3, 2))
    let relK = relativeEmbeddings(w[p + ".emb_rel_k"], T)
    scores = scores + relativeToAbsolute(matmul(q, relK.transposed(0, 2, 1).expandedDimensions(axis: 0)))
    scores = which(mask .== Float(0), MLXArray(Float(-1e4)).asType(scores.dtype), scores)
    let probs = softmax(scores.asType(.float32), axis: -1).asType(v.dtype)
    var out = matmul(probs, v)
    let relV = relativeEmbeddings(w[p + ".emb_rel_v"], T)
    out = out + matmul(absoluteToRelative(probs), relV.expandedDimensions(axis: 0))
    return conv1d(w, p + ".conv_o", out.transposed(0, 2, 1, 3).reshaped(B, T, C))
}

private func ffn(_ w: Weights, _ p: String, _ input: MLXArray, _ mask: MLXArray, _ kernel: Int) -> MLXArray {
    let left = (kernel - 1) / 2, right = kernel / 2
    func same(_ t: MLXArray) -> MLXArray { kernel == 1 ? t : padAxis(t, axis: 1, before: left, after: right) }
    var x = relu(conv1d(w, p + ".conv_1", same(input * mask)))
    x = conv1d(w, p + ".conv_2", same(x * mask))
    return x * mask
}

private func textEncoder(_ w: Weights, _ cfg: SynthConfig, _ input: SynthInputs) -> (m: MLXArray, logs: MLXArray, mask: MLXArray) {
    var x = linear(w, "enc_p.emb_phone", input.phone)
    if cfg.f0, let pitch = input.pitch { x = x + take(w["enc_p.emb_pitch.weight"], pitch, axis: 0) }
    x = leakyRelu(x * Float(cfg.hiddenChannels).squareRoot(), 0.1)
    let T = x.dim(1)
    let mask = MLXArray((0 ..< T).map { $0 < input.length ? Float(1) : Float(0) }, [1, T, 1]).asType(x.dtype)
    let m2 = mask.reshaped(1, 1, T)
    let attnMask = m2.expandedDimensions(axis: -1) * m2.expandedDimensions(axis: -2)   // (1, 1, T, T)

    x = x * mask
    for i in 0 ..< cfg.nLayers {
        let a = attention(w, "enc_p.encoder.attn_layers.\(i)", x, attnMask, cfg.nHeads)
        x = layerNorm(x + a, w["enc_p.encoder.norm_layers_1.\(i).gamma"], w["enc_p.encoder.norm_layers_1.\(i).beta"])
        let f = ffn(w, "enc_p.encoder.ffn_layers.\(i)", x, mask, cfg.kernelSize)
        x = layerNorm(x + f, w["enc_p.encoder.norm_layers_2.\(i).gamma"], w["enc_p.encoder.norm_layers_2.\(i).beta"])
    }
    x = x * mask
    let stats = conv1d(w, "enc_p.proj", x) * mask
    let c = cfg.interChannels
    return (lastSlice(stats, 0, c), lastSlice(stats, c, 2 * c), mask)
}

// MARK: flow

private func wavenet(_ w: Weights, _ p: String, _ input: MLXArray, _ mask: MLXArray, _ g: MLXArray, hidden: Int, layers: Int) -> MLXArray {
    var x = input
    var output = MLXArray.zeros(like: x)
    let cond = conv1d(w, p + ".cond_layer", g)
    for i in 0 ..< layers {
        var xin = conv1d(w, p + ".in_layers.\(i)", x, padding: 2)              // kernel 5, dilation 1
        xin = xin + lastSlice(cond, i * 2 * hidden, (i + 1) * 2 * hidden)
        let acts = tanh(lastSlice(xin, 0, hidden)) * sigmoid(lastSlice(xin, hidden, 2 * hidden))
        let rs = conv1d(w, p + ".res_skip_layers.\(i)", acts)
        if i < layers - 1 {
            x = (x + lastSlice(rs, 0, hidden)) * mask
            output = output + lastSlice(rs, hidden, 2 * hidden)
        } else {
            output = output + rs
        }
    }
    return output * mask
}

private func flowReverse(_ w: Weights, _ cfg: SynthConfig, _ input: MLXArray, _ mask: MLXArray, _ g: MLXArray) -> MLXArray {
    let half = cfg.interChannels / 2
    var x = input
    for f in stride(from: 3, through: 0, by: -1) {                           // flows.{6,4,2,0}, each after a flip
        x = flipAxis(x, axis: -1)
        let p = "flow.flows.\(2 * f)"
        let x0 = lastSlice(x, 0, half), x1 = lastSlice(x, half, 2 * half)
        let h = wavenet(w, p + ".enc", conv1d(w, p + ".pre", x0) * mask, mask, g, hidden: cfg.hiddenChannels, layers: 3)
        let m = conv1d(w, p + ".post", h) * mask
        x = concatenated([x0, (x1 - m) * mask], axis: -1)
    }
    return x
}

// MARK: vocoder

private func resblock(_ w: Weights, _ p: String, _ input: MLXArray, _ cfg: SynthConfig, kernel: Int, dilations: [Int]) -> MLXArray {
    func pad(_ d: Int) -> Int { (kernel * d - d) / 2 }
    var x = input
    if cfg.resblock == "1" {
        for (j, d) in dilations.enumerated() {
            var xt = conv1d(w, p + ".convs1.\(j)", leakyRelu(x, lreluSlope), padding: pad(d), dilation: d)
            xt = conv1d(w, p + ".convs2.\(j)", leakyRelu(xt, lreluSlope), padding: pad(1))
            x = x + xt
        }
    } else {
        for (j, d) in dilations.enumerated() {
            x = x + conv1d(w, p + ".convs.\(j)", leakyRelu(x, lreluSlope), padding: pad(d), dilation: d)
        }
    }
    return x
}

/// NSF sine excitation (SineGen + noise): (1, T) Hz -> (1, T * upp, 1).
func sineExcitation(_ f0In: MLXArray, upp: Int, sr: Int, noise noiseIn: MLXArray?, noiseScale: Float) -> MLXArray {
    let sineAmp: Float = 0.1, noiseStd: Float = 0.003
    let B = f0In.dim(0), T = f0In.dim(1)
    let f0 = f0In.asType(.float32).expandedDimensions(axis: -1)                       // (B, T, 1)
    var rad = f0 * (Float(1) / Float(sr)) * MLXArray((1 ... upp).map { Float($0) }, [upp])  // (B, T, upp)
    var frameEnd = rad[0..., 0 ..< (T - 1), (upp - 1) ..< upp] + Float(0.5)
    frameEnd = frameEnd - floor(frameEnd) - Float(0.5)                                // fmod(x + .5, 1) - .5
    var acc = cumsum(frameEnd, axis: 1)
    acc = acc - which(acc .>= Float(0), floor(acc), ceil(acc))                        // fmod(acc, 1)
    rad = rad + padAxis(acc, axis: 1, before: 1, after: 0)
    let sines = sin(rad.reshaped(B, T * upp, 1) * Float(2 * Double.pi)) * sineAmp
    let uv = repeated((f0 .> Float(0)).asType(.float32), count: upp, axis: 1)
    var noise = noiseIn ?? (noiseScale != 0 ? gaussianNoise(sines.shape) : MLXArray.zeros(sines.shape, dtype: .float32))
    noise = noise * noiseScale * (uv * noiseStd + (1 - uv) * (sineAmp / 3))
    return sines * uv + noise
}

private func generator(_ w: Weights, _ cfg: SynthConfig, _ input: MLXArray, _ g: MLXArray, _ inputs: SynthInputs) -> MLXArray {
    var har: MLXArray? = nil
    if cfg.f0, let pitchf = inputs.pitchf {
        let sine = sineExcitation(pitchf, upp: cfg.hopLength, sr: cfg.sr, noise: inputs.nsfNoise, noiseScale: inputs.nsfNoiseScale)
        har = tanh(linear(w, "dec.m_source.l_linear", sine.asType(input.dtype)))       // (1, T*upp, 1)
    }
    var x = conv1d(w, "dec.conv_pre", input, padding: 3)
    x = x + conv1d(w, "dec.cond", g)
    let nk = cfg.resblockKernelSizes.count
    for (i, (u, k)) in zip(cfg.upsampleRates, cfg.upsampleKernelSizes).enumerated() {
        x = convTranspose1d(w, "dec.ups.\(i)", leakyRelu(x, lreluSlope), stride: u, padding: (k - u) / 2)
        if let har {
            let last = i + 1 == cfg.upsampleRates.count
            let s = cfg.upsampleRates[(i + 1)...].reduce(1, *)
            x = x + conv1d(w, "dec.noise_convs.\(i)", har, stride: last ? 1 : s, padding: last ? 0 : s / 2)
        }
        var xs: MLXArray? = nil
        for j in 0 ..< nk {
            let r = resblock(w, "dec.resblocks.\(i * nk + j)", x, cfg, kernel: cfg.resblockKernelSizes[j],
                             dilations: cfg.resblockDilationSizes[j])
            xs = xs.map { $0 + r } ?? r
        }
        x = xs! * (Float(1) / Float(nk))
    }
    x = MLX.conv1d(leakyRelu(x, 0.01), w["dec.conv_post.weight"], padding: 3)
    return tanh(x)
}

/// RVC `infer`: returns (1, T * hop) audio.
func synthesize(_ w: Weights, _ cfg: SynthConfig, _ inputs: SynthInputs) -> MLXArray {
    let g = inputs.speaker.expandedDimensions(axis: 1)                                 // (1, 1, gin)
    let prior = textEncoder(w, cfg, inputs)
    let eps = inputs.priorNoise ?? (inputs.noiseScale != 0 ? gaussianNoise(prior.m.shape).asType(prior.m.dtype)
                                                            : MLXArray.zeros(like: prior.m))
    let zp = (prior.m + exp(prior.logs) * eps * inputs.noiseScale) * prior.mask
    let z = flowReverse(w, cfg, zp, prior.mask, g)
    let audio = generator(w, cfg, z * prior.mask, g, inputs)
    return audio.squeezed(axis: -1)
}
