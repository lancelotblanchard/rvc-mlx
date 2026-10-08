import Foundation
import MLX

/// HuBERT / ContentVec content encoder (port of rvc_mlx/hubert.py and engine/src/hubert.cpp).
final class Hubert {
    private let w: Weights
    private var convLayers: [(dim: Int, kernel: Int, stride: Int)] =
        [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]
    private var numHeads = 12, numLayers = 12, convPos = 128, convPosGroups = 16
    private let hasFinalProj: Bool
    let dtype: DType

    init(_ weights: Weights) throws {
        w = weights
        let cfg = w.json("config")
        if let layers = cfg["conv_layers"] as? [[Int]] {
            convLayers = layers.map { ($0[0], $0[1], $0[2]) }
            numHeads = cfg["num_heads"] as? Int ?? 12
            numLayers = cfg["num_layers"] as? Int ?? 12
            convPos = cfg["conv_pos"] as? Int ?? 128
            convPosGroups = cfg["conv_pos_groups"] as? Int ?? 16
        }
        hasFinalProj = (w.metadata["has_final_proj"] ?? "1") == "1"
        try w.require(["post_extract_proj.weight", "encoder.pos_conv.weight", "encoder.layer_norm.weight"])
        dtype = w["post_extract_proj.weight"].dtype
    }

    /// RVC content features: v1 = layer 9 + final_proj (256-d), v2 = layer 12 (768-d). audio: (1, L) at 16 kHz.
    func features(_ audio: MLXArray, version: String) throws -> MLXArray {
        if version == "v1" {
            guard hasFinalProj else { throw RVCError.badConfig("this content encoder can't drive v1 voices (no final_proj)") }
            return linear(w, "final_proj", extract(audio, outputLayer: 9))
        }
        return extract(audio, outputLayer: 12)
    }

    func extract(_ audio: MLXArray, outputLayer: Int) -> MLXArray {
        var x = audio.asType(dtype).expandedDimensions(axis: -1)  // (1, L, 1)
        for (i, layer) in convLayers.enumerated() {
            let p = "feature_extractor.conv_layers.\(i)"
            x = MLX.conv1d(x, w[p + ".conv.weight"], stride: layer.stride)
            if i == 0 {  // per-channel GroupNorm over time, in fp32
                let xf = x.asType(.float32)
                let mean = xf.mean(axis: 1, keepDims: true)
                let variance = xf.variance(axis: 1, keepDims: true)
                let normed = (xf - mean) * rsqrt(variance + Float(1e-5))
                x = (normed * w[p + ".norm.weight"].asType(.float32) + w[p + ".norm.bias"].asType(.float32)).asType(dtype)
            }
            x = gelu(x)
        }
        x = linear(w, "post_extract_proj", layerNorm(w, "layer_norm", x))

        var pos = conv1d(w, "encoder.pos_conv", x, padding: convPos / 2, groups: convPosGroups)
        if convPos % 2 == 0 { pos = timeSlice(pos, 0, pos.dim(1) - 1) }  // SamePad
        x = layerNorm(w, "encoder.layer_norm", x + gelu(pos))

        let B = x.dim(0), T = x.dim(1), C = x.dim(2), H = numHeads, D = C / H
        let scale = Float(1) / Float(D).squareRoot()
        func heads(_ t: MLXArray) -> MLXArray { t.reshaped(B, T, H, D).transposed(0, 2, 1, 3) }
        for i in 0 ..< min(outputLayer, numLayers) {
            let p = "encoder.layers.\(i)"
            let q = heads(linear(w, p + ".self_attn.q_proj", x))
            let k = heads(linear(w, p + ".self_attn.k_proj", x))
            let v = heads(linear(w, p + ".self_attn.v_proj", x))
            let scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
            let probs = softmax(scores.asType(.float32), axis: -1).asType(v.dtype)
            let o = matmul(probs, v).transposed(0, 2, 1, 3).reshaped(B, T, C)
            x = layerNorm(w, p + ".self_attn_layer_norm", x + linear(w, p + ".self_attn.out_proj", o))
            let ff = linear(w, p + ".fc2", gelu(linear(w, p + ".fc1", x)))
            x = layerNorm(w, p + ".final_layer_norm", x + ff)
        }
        return x
    }
}
