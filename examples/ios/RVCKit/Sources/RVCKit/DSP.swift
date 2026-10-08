import Foundation

// Port of engine/src/dsp.cpp (scipy / numpy / librosa semantics, checked against them in the C++ parity tests).

private let hpB: [Double] = [0.9699606451838447, -4.849803225919223, 9.699606451838447,
                             -9.699606451838447, 4.849803225919223, -0.9699606451838447]
private let hpA: [Double] = [1.0, -4.939001819168364, 9.757863526739543,
                             -9.639544849413458, 4.761506797356209, -0.9408236532054606]
private let hpZi: [Double] = [-0.9699607413707367, 3.879842959615721, -5.81976443080129,
                              3.879842948235015, -0.9699607356787477]

private func lfilter(_ x: [Double], initScale: Double) -> [Double] {
    var z = hpZi.map { $0 * initScale }
    var y = [Double](repeating: 0, count: x.count)
    for n in 0 ..< x.count {
        let xn = x[n]
        let yn = hpB[0] * xn + z[0]
        for i in 0 ..< 4 { z[i] = hpB[i + 1] * xn + z[i + 1] - hpA[i + 1] * yn }
        z[4] = hpB[5] * xn - hpA[5] * yn
        y[n] = yn
    }
    return y
}

/// scipy.signal.filtfilt with RVC's 48 Hz Butterworth high-pass.
func highpass(_ x: [Double]) -> [Double] {
    let padlen = 18, n = x.count
    precondition(n > padlen)
    var ext = [Double]()
    ext.reserveCapacity(n + 2 * padlen)
    for i in stride(from: padlen, through: 1, by: -1) { ext.append(2 * x[0] - x[i]) }
    ext.append(contentsOf: x)
    for i in 2 ... (padlen + 1) { ext.append(2 * x[n - 1] - x[n - i]) }
    var y = lfilter(ext, initScale: ext[0])
    y.reverse()
    y = lfilter(y, initScale: y[0])
    y.reverse()
    return Array(y[padlen ..< (y.count - padlen)])
}

func reflectPad(_ x: [Double], left: Int, right: Int) -> [Double] {
    reflectIndices(x.count, left: left, right: right).map { x[$0] }
}

func frameRms(_ y: [Double], frameLength: Int, hopLength: Int) -> [Double] {
    let pad = frameLength / 2, len = y.count + 2 * pad
    guard len >= frameLength else { return [0] }
    var prefix = [Double](repeating: 0, count: len + 1)
    for i in 0 ..< len {
        let v = (i >= pad && i - pad < y.count) ? y[i - pad] : 0
        prefix[i + 1] = prefix[i] + v * v
    }
    let frames = 1 + (len - frameLength) / hopLength
    return (0 ..< frames).map { f in
        let s = f * hopLength
        return (max(0, prefix[s + frameLength] - prefix[s]) / Double(frameLength)).squareRoot()
    }
}

func interpLinear(_ x: [Double], size: Int) -> [Double] {
    let n = Double(x.count)
    return (0 ..< size).map { i in
        let pos = min(max((Double(i) + 0.5) * (n / Double(size)) - 0.5, 0), n - 1)
        let lo = Int(pos.rounded(.down)), hi = min(lo + 1, x.count - 1)
        let frac = pos - Double(lo)
        return x[lo] * (1 - frac) + x[hi] * frac
    }
}

/// RVC's loudness-envelope mix.
func changeRms(source: [Double], sourceRate: Int, target: inout [Float], targetRate: Int, rate: Double) {
    let t = target.map { Double($0) }
    let rms1 = interpLinear(frameRms(source, frameLength: sourceRate / 2 * 2, hopLength: sourceRate / 2), size: target.count)
    let rms2 = interpLinear(frameRms(t, frameLength: targetRate / 2 * 2, hopLength: targetRate / 2), size: target.count)
    for i in 0 ..< target.count {
        target[i] = Float(t[i] * pow(rms1[i], 1 - rate) * pow(max(rms2[i], 1e-6), rate - 1))
    }
}

func coarseF0(_ f0: [Double]) -> [Int] {
    let melMin = 1127 * log(1 + 50.0 / 700), melMax = 1127 * log(1 + 1100.0 / 700)
    return f0.map { hz in
        var mel = 1127 * log(1 + hz / 700)
        if mel > 0 { mel = (mel - melMin) * 254 / (melMax - melMin) + 1 }
        return Int(min(max(mel, 1), 255).rounded(.toNearestOrEven))
    }
}

func splitPoints(_ audio: [Double], window: Int, tMax: Int, tCenter: Int, tQuery: Int) -> [Int] {
    let pad = reflectPad(audio, left: window / 2, right: window / 2)
    guard pad.count > tMax else { return [] }
    let n = audio.count
    var sum = [Double](repeating: 0, count: n)
    for i in 0 ..< window {
        for j in 0 ..< n { sum[j] += abs(pad[i + j]) }
    }
    var ts: [Int] = []
    var t = tCenter
    while t < n {
        let lo = t - tQuery, hi = min(n, t + tQuery)
        var best = lo
        for j in lo ..< hi where sum[j] < sum[best] { best = j }
        ts.append(best)
        t += tCenter
    }
    return ts
}

// MARK: - Resampling

private func besselI0(_ x: Double) -> Double {
    var sum = 1.0, term = 1.0
    for k in 1 ..< 50 {
        term *= (x / Double(2 * k)) * (x / Double(2 * k))
        sum += term
        if term < 1e-12 * sum { break }
    }
    return sum
}

/// Band-limited (Kaiser-windowed sinc) sample-rate conversion.
public func resample(_ input: [Float], from fromRate: Double, to toRate: Double) -> [Float] {
    if fromRate == toRate || input.isEmpty { return input }
    let zeroCrossings = 32, tableRes = 512, beta = 9.0
    let ratio = toRate / fromRate
    let cutoff = min(1.0, ratio) * 0.95
    let span = Double(zeroCrossings) / cutoff
    let tableSize = Int((span * Double(tableRes)).rounded(.up)) + 2
    let i0beta = besselI0(beta)
    let table: [Double] = (0 ..< tableSize).map { i in
        let u = Double(i) / Double(tableRes), r = u / span
        let win = r >= 1 ? 0 : besselI0(beta * (1 - r * r).squareRoot()) / i0beta
        let x = Double.pi * cutoff * u
        return cutoff * (u == 0 ? 1 : sin(x) / x) * win
    }
    let n = input.count
    let outLen = Int((Double(n) * ratio).rounded(.up))
    let reach = Int(span.rounded(.up))
    var out = [Float](repeating: 0, count: outLen)
    input.withUnsafeBufferPointer { src in
        table.withUnsafeBufferPointer { h in
            for o in 0 ..< outLen {
                let t = Double(o) / ratio
                let c = Int(t.rounded(.down))
                var acc = 0.0
                var j = max(0, c - reach + 1)
                let jEnd = min(n - 1, c + reach)
                while j <= jEnd {
                    let u = abs(t - Double(j)) * Double(tableRes)
                    let k = Int(u)
                    if k + 1 < h.count {
                        let f = u - Double(k)
                        acc += Double(src[j]) * (h[k] + (h[k + 1] - h[k]) * f)
                    }
                    j += 1
                }
                out[o] = Float(acc)
            }
        }
    }
    return out
}
