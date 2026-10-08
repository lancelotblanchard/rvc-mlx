import Foundation
import MLX

public enum RVCError: LocalizedError {
    case unreadable(String, String)
    case wrongKind(file: String, kind: String, expected: String)
    case missingTensor(String, file: String)
    case badConfig(String)
    case audioTooShort
    case incompatibleVoices(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .unreadable(let file, let why): return "Couldn't read \(file): \(why)"
        case .wrongKind(let file, let kind, let expected): return "\(file) is a \(kind) file, expected \(expected)."
        case .missingTensor(let key, let file): return "\(file) has no tensor \(key). Re-convert it with rvc-mlx."
        case .badConfig(let why): return "Invalid model config: \(why)"
        case .audioTooShort: return "The recording is too short (at least 0.1 s is needed)."
        case .incompatibleVoices(let why): return "These voices can't be blended: \(why)"
        case .cancelled: return "Conversion cancelled."
        }
    }
}

public enum Precision: Sendable {
    case float32, float16

    var dtype: DType { self == .float16 ? .float16 : .float32 }
}

/// A converted `.safetensors` file: MLX-layout tensors plus the rvc-mlx metadata header.
struct Weights {
    var arrays: [String: MLXArray]
    var metadata: [String: String]
    let file: String

    /// Floating tensors are cast to `dtype`, except names in `keepFloat32` (front-end constants).
    static func load(_ url: URL, kind: String, dtype: DType, keepFloat32: Set<String> = []) throws -> Weights {
        let loaded: ([String: MLXArray], [String: String])
        do {
            loaded = try loadArraysAndMetadata(url: url)
        } catch {
            throw RVCError.unreadable(url.lastPathComponent, error.localizedDescription)
        }
        if let found = loaded.1["kind"], found != kind {
            throw RVCError.wrongKind(file: url.lastPathComponent, kind: found, expected: kind)
        }
        var arrays: [String: MLXArray] = [:]
        for (name, array) in loaded.0 {
            let floating = [DType.float16, .float32, .bfloat16].contains(array.dtype)
            arrays[name] = floating ? array.asType(keepFloat32.contains(name) ? .float32 : dtype) : array
        }
        eval(Array(arrays.values))
        return Weights(arrays: arrays, metadata: loaded.1, file: url.lastPathComponent)
    }

    func has(_ key: String) -> Bool { arrays[key] != nil }

    /// Required tensor. Files are validated by `require` at load time, so a miss here is a programming error.
    subscript(_ key: String) -> MLXArray {
        guard let a = arrays[key] else { fatalError("\(file) has no tensor \(key)") }
        return a
    }

    func require(_ keys: [String]) throws {
        for k in keys where arrays[k] == nil { throw RVCError.missingTensor(k, file: file) }
    }

    func json(_ key: String) -> [String: Any] {
        guard let text = metadata[key], let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return object
    }
}

// MARK: - Functional layers (channels-last, mirroring the C++ engine)

func linear(_ w: Weights, _ name: String, _ x: MLXArray) -> MLXArray {
    var y = matmul(x, w[name + ".weight"].T)
    if let b = w.arrays[name + ".bias"] { y = y + b }
    return y
}

/// x: (B, T, Cin), weight: (Cout, K, Cin / groups).
func conv1d(_ w: Weights, _ name: String, _ x: MLXArray, stride: Int = 1, padding: Int = 0, dilation: Int = 1,
            groups: Int = 1) -> MLXArray {
    var y = MLX.conv1d(x, w[name + ".weight"], stride: stride, padding: padding, dilation: dilation, groups: groups)
    if let b = w.arrays[name + ".bias"] { y = y + b }
    return y
}

/// PyTorch-semantics transposed convolution: x (B, T, Cin), weight (Cout, K, Cin).
func convTranspose1d(_ w: Weights, _ name: String, _ x: MLXArray, stride: Int, padding: Int) -> MLXArray {
    var y = MLX.convTransposed1d(x, w[name + ".weight"], stride: stride, padding: padding)
    if let b = w.arrays[name + ".bias"] { y = y + b }
    return y
}

/// x: (B, H, W, Cin), weight: (Cout, KH, KW, Cin), stride 1, "same" padding for 3x3 (1) or none for 1x1 (0).
func conv2d(_ w: Weights, _ name: String, _ x: MLXArray, padding: Int) -> MLXArray {
    precondition(padding == 0 || padding == 1)
    var y = padding == 1 ? MLX.conv2d(x, w[name + ".weight"], stride: 1, padding: 1)
                         : MLX.conv2d(x, w[name + ".weight"], stride: 1, padding: 0)
    if let b = w.arrays[name + ".bias"] { y = y + b }
    return y
}

func layerNorm(_ x: MLXArray, _ gamma: MLXArray, _ beta: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let x32 = x.asType(.float32)
    let mean = x32.mean(axis: -1, keepDims: true)
    let variance = x32.variance(axis: -1, keepDims: true)
    let y = (x32 - mean) * rsqrt(variance + eps) * gamma.asType(.float32) + beta.asType(.float32)
    return y.asType(x.dtype)
}

func layerNorm(_ w: Weights, _ name: String, _ x: MLXArray) -> MLXArray {
    layerNorm(x, w[name + ".weight"], w[name + ".bias"])
}

/// Inference BatchNorm over the last (channel) axis.
func batchNorm(_ w: Weights, _ name: String, _ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let scale = w[name + ".weight"] * rsqrt(w[name + ".running_var"] + eps)
    return (x - w[name + ".running_mean"]) * scale + w[name + ".bias"]
}

func gelu(_ x: MLXArray) -> MLXArray { x * 0.5 * (1 + erf(x * Float(0.70710678118654752))) }

func leakyRelu(_ x: MLXArray, _ slope: Float) -> MLXArray { maximum(x, x * slope) }

func relu(_ x: MLXArray) -> MLXArray { maximum(x, MLXArray(Float(0)).asType(x.dtype)) }

/// x[..., start ..< end]
func lastSlice(_ x: MLXArray, _ start: Int, _ end: Int) -> MLXArray { x[.ellipsis, start ..< end] }

/// x[:, start ..< end] (the time axis of (B, T, ...) tensors)
func timeSlice(_ x: MLXArray, _ start: Int, _ end: Int) -> MLXArray { x[0..., start ..< end] }

/// Zero padding along `axis`.
func padAxis(_ x: MLXArray, axis: Int, before: Int, after: Int) -> MLXArray {
    let ax = axis < 0 ? axis + x.ndim : axis
    var parts: [MLXArray] = []
    if before > 0 {
        var s = x.shape
        s[ax] = before
        parts.append(MLXArray.zeros(s, dtype: x.dtype))
    }
    parts.append(x)
    if after > 0 {
        var s = x.shape
        s[ax] = after
        parts.append(MLXArray.zeros(s, dtype: x.dtype))
    }
    return parts.count == 1 ? x : concatenated(parts, axis: ax)
}

func flipAxis(_ x: MLXArray, axis: Int) -> MLXArray {
    let ax = axis < 0 ? axis + x.ndim : axis
    let n = x.dim(ax)
    return take(x, MLXArray((0 ..< n).reversed().map { Int32($0) }), axis: ax)
}

func int32Array(_ values: [Int], _ shape: [Int]) -> MLXArray { MLXArray(values.map { Int32($0) }, shape) }

/// Standard normal samples (Box–Muller), generated on the CPU so RVCKit needs no extra MLX products.
func gaussianNoise(_ shape: [Int]) -> MLXArray {
    let count = shape.reduce(1, *)
    var out = [Float](repeating: 0, count: count)
    var i = 0
    while i < count {
        let u1 = Float.random(in: Float.ulpOfOne ..< 1), u2 = Float.random(in: 0 ..< 1)
        let r = (-2 * log(u1)).squareRoot()
        out[i] = r * cos(2 * .pi * u2)
        if i + 1 < count { out[i + 1] = r * sin(2 * .pi * u2) }
        i += 2
    }
    return MLXArray(out, shape)
}
