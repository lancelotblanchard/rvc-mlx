import Foundation
import MLX
import XCTest

@testable import RVCKit

/// Checks RVCKit against outputs of the Python implementation (the same golden files as the C++ engine's test).
///
///     python examples/juce-plugin/engine/tests/make_golden.py /tmp/rvc-golden
///     RVC_GOLDEN_DIR=/tmp/rvc-golden swift test        # on an Apple-silicon Mac
final class ParityTests: XCTestCase {
    private var dir: URL!
    private var golden: [String: MLXArray] = [:]

    override func setUpWithError() throws {
        guard let path = ProcessInfo.processInfo.environment["RVC_GOLDEN_DIR"] else {
            throw XCTSkip("Set RVC_GOLDEN_DIR to the output of make_golden.py")
        }
        dir = URL(fileURLWithPath: path)
        golden = try loadArrays(url: dir.appendingPathComponent("golden.safetensors"))
    }

    private func g(_ key: String) -> MLXArray { golden[key]! }
    private func floats(_ a: MLXArray) -> [Float] { a.asType(.float32).reshaped(-1).asArray(Float.self) }
    private func url(_ name: String) -> URL { dir.appendingPathComponent(name) }

    private func assertClose(_ got: [Float], _ ref: [Float], tol: Double, _ name: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, ref.count, "\(name): size", file: file, line: line)
        guard got.count == ref.count else { return }
        var maxErr = 0.0, maxRef = 1e-9
        for (a, b) in zip(got, ref) {
            maxErr = max(maxErr, abs(Double(a) - Double(b)))
            maxRef = max(maxRef, abs(Double(b)))
        }
        XCTAssertLessThanOrEqual(maxErr / maxRef, tol, "\(name): relative error \(maxErr / maxRef)", file: file, line: line)
    }

    private func assertClose(_ got: MLXArray, _ ref: MLXArray, tol: Double, _ name: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.shape, ref.shape, "\(name): shape", file: file, line: line)
        assertClose(floats(got), floats(ref), tol: tol, name, file: file, line: line)
    }

    private var audio: [Float] { floats(g("audio")) }

    func testHubert() throws {
        let hubert = try Hubert(Weights.load(url("hubert.safetensors"), kind: "hubert", dtype: .float32))
        let x = MLXArray(Array(audio.prefix(32000)), [1, 32000])
        assertClose(try hubert.features(x, version: "v2"), g("hubert_v2"), tol: 1e-3, "hubert v2")
        assertClose(try hubert.features(x, version: "v1"), g("hubert_v1"), tol: 1e-3, "hubert v1")
    }

    func testRmvpe() throws {
        let rmvpe = try Rmvpe(Weights.load(url("rmvpe.safetensors"), kind: "rmvpe", dtype: .float32, keepFloat32: ["mel_basis"]))
        let a = Array(audio.prefix(32000))
        assertClose(rmvpe.mel(MLXArray(a, [1, a.count])), g("rmvpe_mel"), tol: 1e-4, "log-mel")
        assertClose(rmvpe.salience(g("rmvpe_mel")), g("rmvpe_salience"), tol: 1e-3, "salience")
        assertClose(rmvpe.f0(a).map { Float($0) }, floats(g("rmvpe_f0")), tol: 1e-3, "f0")
    }

    func testSineExcitation() {
        assertClose(sineExcitation(g("sine_f0"), upp: 160, sr: 16000, noise: g("sine_noise"), noiseScale: 1),
                    g("sine_out"), tol: 2e-3, "sine")
    }

    func testSynthesizers() throws {
        for name in ["v2_f0", "v1_f0", "v2_nof0"] {
            let voice = try Voice.load(url("\(name).safetensors"), precision: .float32)
            let phone = g("synth_\(name)_phone")
            var inputs = SynthInputs(phone: phone, length: phone.dim(1), speaker: voice.weights["emb_g.weight"][2 ..< 3])
            inputs.pitch = g("synth_\(name)_pitch")
            inputs.pitchf = g("synth_\(name)_pitchf")
            inputs.priorNoise = g("synth_\(name)_prior")
            inputs.nsfNoise = g("synth_\(name)_nsf")
            assertClose(synthesize(voice.weights, voice.config, inputs), g("synth_\(name)_out"), tol: 2e-3, name)
        }
    }

    func testPipelines() throws {
        let engine = try RVCEngine(hubertURL: url("hubert.safetensors"), rmvpeURL: url("rmvpe.safetensors"), precision: .float32)
        engine.chunking = ChunkingConfig(xPad: 1, xQuery: 1, xCenter: 3, xMax: 4)
        for name in ["v2_f0", "v1_f0", "v2_nof0"] {
            let voice = try Voice.load(url("\(name).safetensors"), precision: .float32)
            let a = try engine.convert(audio, voice: voice, options: .init(pitch: 3, indexRate: 0.75, protect: 0.33, rmsMixRate: 0.25, speaker: 1, deterministic: true))
            assertClose(a, floats(g("pipe_\(name)_a")), tol: 3e-3, "pipeline \(name) a")
            let b = try engine.convert(audio, voice: voice, options: .init(pitch: -5, indexRate: 0, protect: 0.5, rmsMixRate: 1, speaker: 1, deterministic: true))
            assertClose(b, floats(g("pipe_\(name)_b")), tol: 3e-3, "pipeline \(name) b")
        }
        let voice = try Voice.load(url("v2_f0.safetensors"), precision: .float32)
        let edge = try engine.convert(floats(g("edge_audio")), voice: voice, options: .init(indexRate: 0.5, deterministic: true))
        assertClose(edge, floats(g("pipe_edge")), tol: 3e-3, "split at the last frame")
        XCTAssertThrowsError(try engine.convert(audio, voice: voice, progress: { _ in false }))
    }

    func testBlendAndRetrieval() throws {
        let a = try Voice.load(url("v2_f0.safetensors"), precision: .float32)
        let b = try Voice.load(url("v2_f0_b.safetensors"), precision: .float32)
        let mix = try Voice.blend([a, b], weights: [3, 7])
        for key in ["dec.ups.0.weight", "enc_p.emb_phone.weight", "emb_g.weight", "flow.flows.2.enc.cond_layer.weight"] {
            assertClose(mix.weights[key], g("blend_\(key)"), tol: 1e-6, key)
        }
        XCTAssertEqual(mix.info.mergedFrom, "v2_f0:0.300, v2_f0_b:0.700")
        assertClose(retrieveBlended(g("retrieve_q"), banks: [a.banks[0], b.banks[0]], weights: [0.3, 0.7]),
                    g("retrieve_out"), tol: 1e-4, "retrieval")
        let v1 = try Voice.load(url("v1_f0.safetensors"), precision: .float32)
        XCTAssertNotNil(Voice.blendProblem(a, v1))
    }

    func testDSP() {
        let x = floats(g("dsp_x")).map { Double($0) }
        assertClose(highpass(x).map { Float($0) }, floats(g("dsp_highpass")), tol: 1e-5, "highpass")
        var t = floats(g("dsp_rms_target"))
        changeRms(source: x, sourceRate: 16000, target: &t, targetRate: 40000, rate: 0.25)
        assertClose(t, floats(g("dsp_rms_out")), tol: 1e-4, "change_rms")
        XCTAssertEqual(reflectPad([1, 2, 3], left: 5, right: 4), [2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3])
    }
}

final class ResampleTests: XCTestCase {
    func testRoundTripAndAntiAliasing() {
        let sr = 44100.0
        let tone = (0 ..< 44100).map { Float(0.5 * sin(2 * Double.pi * 440 * Double($0) / sr)) }
        let down = resample(tone, from: sr, to: 16000)
        let back = resample(down, from: 16000, to: sr)
        XCTAssertEqual(down.count, 16000)
        XCTAssertEqual(back.count, 44100)
        let err = (2000 ..< 42000).map { abs(back[$0] - tone[$0]) }.max()!
        XCTAssertLessThan(err, 2e-3)

        let high = (0 ..< 44100).map { Float(0.5 * sin(2 * Double.pi * 10000 * Double($0) / sr)) }
        let aliased = resample(high, from: sr, to: 16000)
        let rms = (aliased[4000 ..< 12000].map { Double($0 * $0) }.reduce(0, +) / 8000).squareRoot()
        XCTAssertLessThan(rms, 1e-3)  // a 10 kHz tone can't survive conversion to 16 kHz
    }

    func testModelKindReadsHeaderOnly() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("kind.safetensors")
        try save(arrays: ["x": MLXArray([Float(1)], [1])], metadata: ["kind": "voice", "name": "Test"], url: tmp)
        let kind = ModelKind.of(tmp)
        XCTAssertEqual(kind?.kind, .voice)
        XCTAssertEqual(kind?.metadata["name"], "Test")
    }
}
