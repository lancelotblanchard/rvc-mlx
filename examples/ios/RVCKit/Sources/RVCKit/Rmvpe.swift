import Foundation
import MLX

/// RMVPE pitch estimator (port of rvc_mlx/rmvpe.py and engine/src/rmvpe.cpp).
final class Rmvpe {
    private static let nFFT = 1024, hop = 160, bins = 360, pad = 4
    private static let centsOffset = 1997.3794084376191, centsStep = 20.0

    private let w: Weights
    private var nBlocks = 4, enDeLayers = 5, interLayers = 4, enOutChannels = 16, nGru = 1
    private let dtype: DType
    private let window: MLXArray
    private let dftCos: MLXArray, dftSin: MLXArray  // (1024, 513): the real DFT as two matmuls

    init(_ weights: Weights) throws {
        w = weights
        let cfg = w.json("config")
        nBlocks = cfg["n_blocks"] as? Int ?? 4
        nGru = cfg["n_gru"] as? Int ?? 1
        enDeLayers = cfg["en_de_layers"] as? Int ?? 5
        interLayers = cfg["inter_layers"] as? Int ?? 4
        enOutChannels = cfg["en_out_channels"] as? Int ?? 16
        try w.require(["mel_basis", "cnn.weight", "linear.weight", "unet.encoder.bn.weight"])
        dtype = w["cnn.weight"].dtype

        let n = Rmvpe.nFFT, half = n / 2 + 1
        window = MLXArray((0 ..< n).map { Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double($0) / Double(n))) }, [n])
        var c = [Float](repeating: 0, count: n * half), s = [Float](repeating: 0, count: n * half)
        for t in 0 ..< n {
            for f in 0 ..< half {
                let angle = 2.0 * Double.pi * Double((t * f) % n) / Double(n)
                c[t * half + f] = Float(cos(angle))
                s[t * half + f] = Float(-sin(angle))
            }
        }
        dftCos = MLXArray(c, [n, half])
        dftSin = MLXArray(s, [n, half])
    }

    /// (1, L) audio -> (1, 128, frames) log-mel spectrogram (centered STFT with reflect padding).
    func mel(_ audio: MLXArray) -> MLXArray {
        let length = audio.shape.last!, half = Rmvpe.nFFT / 2
        let padded = take(audio.asType(.float32), int32Array(reflectIndices(length, left: half, right: half),
                                                              [length + 2 * half]), axis: -1)  // (1, Lp)
        let frames = 1 + (length + 2 * half - Rmvpe.nFFT) / Rmvpe.hop
        var idx = [Int](repeating: 0, count: frames * Rmvpe.nFFT)
        for f in 0 ..< frames {
            for k in 0 ..< Rmvpe.nFFT { idx[f * Rmvpe.nFFT + k] = f * Rmvpe.hop + k }
        }
        let framed = take(padded.reshaped(-1), int32Array(idx, [frames, Rmvpe.nFFT]), axis: 0) * window  // (frames, 1024)
        let re = matmul(framed, dftCos), im = matmul(framed, dftSin)
        let magnitude = sqrt(re * re + im * im)                                                  // (frames, 513)
        let melSpec = matmul(w["mel_basis"].asType(.float32), magnitude.T)                       // (128, frames)
        return log(maximum(melSpec, Float(1e-5))).expandedDimensions(axis: 0)
    }

    private func convBlockRes(_ p: String, _ x: MLXArray, inC: Int, outC: Int) -> MLXArray {
        var y = relu(batchNorm(w, p + ".conv.layers.1", conv2d(w, p + ".conv.layers.0", x, padding: 1)))
        y = relu(batchNorm(w, p + ".conv.layers.4", conv2d(w, p + ".conv.layers.3", y, padding: 1)))
        return y + (inC != outC ? conv2d(w, p + ".shortcut", x, padding: 0) : x)
    }

    private func biGru(_ input: MLXArray) -> MLXArray {
        var hIn = input
        for layer in 0 ..< nGru {
            let f = "gru.forward_grus.\(layer)", b = "gru.backward_grus.\(layer)"
            let T = hIn.dim(1), H = w[f + ".Wh"].dim(1)
            let xf = matmul(hIn, w[f + ".Wx"].T) + w[f + ".b"]
            let xb = matmul(flipAxis(hIn, axis: 1), w[b + ".Wx"].T) + w[b + ".b"]
            let X = stacked([xf, xb], axis: 0)                                         // (2, 1, T, 3H)
            let Wh = stacked([w[f + ".Wh"].T, w[b + ".Wh"].T], axis: 0)                // (2, H, 3H)
            let bhn = stacked([w[f + ".bhn"], w[b + ".bhn"]], axis: 0).reshaped(2, 1, H)
            var h = MLXArray.zeros([2, 1, H], dtype: X.dtype)
            var outs: [MLXArray] = []
            outs.reserveCapacity(T)
            for t in 0 ..< T {
                let xt = X[0..., 0..., t, 0...]                                        // (2, 1, 3H)
                let hp = matmul(h, Wh)
                let rz = sigmoid(lastSlice(xt, 0, 2 * H) + lastSlice(hp, 0, 2 * H))
                let r = lastSlice(rz, 0, H), z = lastSlice(rz, H, 2 * H)
                let n = tanh(lastSlice(xt, 2 * H, 3 * H) + r * (lastSlice(hp, 2 * H, 3 * H) + bhn))
                h = (1 - z) * n + z * h
                outs.append(h)
                if (t + 1) % 512 == 0 { eval(h) }
            }
            let seq = stacked(outs, axis: 2)                                           // (2, 1, T, H)
            hIn = concatenated([seq[0], flipAxis(seq[1], axis: 1)], axis: -1)          // (1, T, 2H)
        }
        return hIn
    }

    /// (1, 128, frames) -> (1, frames, 360) pitch salience.
    func salience(_ melIn: MLXArray) -> MLXArray {
        let frames = melIn.shape.last!
        let padded = 32 * ((frames - 1) / 32 + 1)
        var x = padAxis(melIn, axis: -1, before: 0, after: padded - frames)
            .transposed(0, 2, 1).expandedDimensions(axis: -1).asType(dtype)            // (1, T, 128, 1)

        x = batchNorm(w, "unet.encoder.bn", x)
        var skips: [MLXArray] = []
        var inC = 1, outC = enOutChannels
        for i in 0 ..< enDeLayers {
            for j in 0 ..< nBlocks {
                x = convBlockRes("unet.encoder.layers.\(i).conv.\(j)", x, inC: j == 0 ? inC : outC, outC: outC)
            }
            skips.append(x)
            let B = x.dim(0), H = x.dim(1), W = x.dim(2), C = x.dim(3)
            x = x.reshaped(B, H / 2, 2, W / 2, 2, C).mean(axes: [2, 4])               // AvgPool2d(2)
            inC = outC
            outC *= 2
        }
        let top = outC
        for i in 0 ..< interLayers {
            let cin = i == 0 ? top / 2 : top
            for j in 0 ..< nBlocks {
                x = convBlockRes("unet.intermediate.layers.\(i).conv.\(j)", x, inC: j == 0 ? cin : top, outC: top)
            }
        }
        var c = top
        for i in 0 ..< enDeLayers {
            let p = "unet.decoder.layers.\(i)", out = c / 2
            // ConvTranspose2d(k=3, s=2, p=1, output_padding=1) == full transposed conv cropped to [1, 2H + 1).
            let H = x.dim(1), W = x.dim(2)
            let full = MLX.convTransposed2d(x, w[p + ".conv1.layers.0.weight"], stride: 2, padding: 0)
            x = full[0..., 1 ..< (2 * H + 1), 1 ..< (2 * W + 1), 0...]
            x = relu(batchNorm(w, p + ".conv1.layers.1", x))
            x = concatenated([x, skips[skips.count - 1 - i]], axis: -1)
            for j in 0 ..< nBlocks {
                x = convBlockRes(p + ".conv2.\(j)", x, inC: j == 0 ? out * 2 : out, outC: out)
            }
            c = out
        }
        x = conv2d(w, "cnn", x, padding: 1)                                            // (1, T, 128, 3)
        x = x.transposed(0, 1, 3, 2)
        x = x.reshaped(x.dim(0), x.dim(1), -1)                                         // (1, T, 384)
        x = sigmoid(linear(w, "linear", biGru(x)))
        return timeSlice(x, 0, frames)
    }

    /// f0 in Hz (0 = unvoiced), one value per 160 samples.
    func f0(_ audio16k: [Float], threshold: Float = 0.03) -> [Double] {
        let audio = MLXArray(audio16k, [1, audio16k.count])
        let s = salience(mel(audio)).asType(.float32)
        eval(s)
        let T = s.dim(1)
        let sal = s.asArray(Float.self)
        var out = [Double](repeating: 0, count: T)
        for t in 0 ..< T {
            let row = t * Rmvpe.bins
            var center = 0
            var peak = -Float.infinity
            for k in 0 ..< Rmvpe.bins where sal[row + k] > peak { peak = sal[row + k]; center = k }
            var num = 0.0, den = 0.0
            for k in max(0, center - Rmvpe.pad) ... min(Rmvpe.bins - 1, center + Rmvpe.pad) {
                num += Double(sal[row + k]) * (Rmvpe.centsStep * Double(k) + Rmvpe.centsOffset)
                den += Double(sal[row + k])
            }
            if peak <= threshold || den == 0 { continue }
            let hz = 10.0 * pow(2.0, (num / den) / 1200.0)
            out[t] = hz == 10.0 ? 0 : hz
        }
        return out
    }
}

/// numpy-style reflect padding indices (valid for pads longer than the signal).
func reflectIndices(_ n: Int, left: Int, right: Int) -> [Int] {
    let period = max(1, 2 * (n - 1))
    return (-left ..< (n + right)).map { i in
        if n == 1 { return 0 }
        var j = abs(i) % period
        if j >= n { j = period - j }
        return j
    }
}
