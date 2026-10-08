import Foundation
import RVCKit

/// The converted models on this device. Everything lives in the app's Documents folder, which is visible in the
/// Files app ("On My iPhone › RVC Pocket"), so models can also be dropped in from a Mac.
@MainActor
final class ModelLibrary: ObservableObject {
    struct VoiceItem: Identifiable, Hashable {
        let url: URL
        let name: String
        let detail: String
        var id: URL { url }
    }

    @Published private(set) var hubertURL: URL?
    @Published private(set) var rmvpeURL: URL?
    @Published private(set) var voices: [VoiceItem] = []
    @Published var importMessage: String?

    var isReady: Bool { hubertURL != nil && rmvpeURL != nil && !voices.isEmpty }

    private let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    private var modelsDir: URL { root.appendingPathComponent("Models", isDirectory: true) }
    private var voicesDir: URL { root.appendingPathComponent("Voices", isDirectory: true) }

    init() {
        try? FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        refresh()
    }

    /// Scans Documents (recursively) and classifies every .safetensors by its header.
    func refresh() {
        var hubert: URL?, rmvpe: URL?, voiceItems: [VoiceItem] = []
        let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }.filter { $0.pathExtension == "safetensors" } ?? []
        for url in files.sorted(by: { $0.path < $1.path }) {
            guard let found = ModelKind.of(url) else { continue }
            let meta = found.metadata
            switch found.kind {
            case .hubert: hubert = hubert ?? url
            case .rmvpe: rmvpe = rmvpe ?? url
            case .voice:
                let rate = (Int(meta["sample_rate"] ?? "") ?? 0) / 1000
                var detail = "\(meta["version"] ?? "v2") · \(rate) kHz"
                if (Int(meta["index_size"] ?? "0") ?? 0) > 0 { detail += " · index" }
                if meta["merged_from"] != nil { detail += " · blend" }
                voiceItems.append(VoiceItem(url: url, name: meta["name"] ?? url.deletingPathExtension().lastPathComponent, detail: detail))
            }
        }
        hubertURL = hubert
        rmvpeURL = rmvpe
        voices = voiceItems.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    /// Copies picked / AirDropped files into the library. Non-model files are reported, not imported.
    func importFiles(_ urls: [URL]) {
        var imported: [String] = [], rejected: [String] = []
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            guard let kind = ModelKind.of(url)?.kind else {
                rejected.append(url.lastPathComponent)
                continue
            }
            let destination: URL
            switch kind {
            case .hubert: destination = modelsDir.appendingPathComponent("hubert.safetensors")
            case .rmvpe: destination = modelsDir.appendingPathComponent("rmvpe.safetensors")
            case .voice: destination = voicesDir.appendingPathComponent(url.lastPathComponent)
            }
            if destination.standardizedFileURL == url.standardizedFileURL {
                imported.append(url.lastPathComponent)
                continue
            }
            do {
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.copyItem(at: url, to: destination)
                imported.append(url.lastPathComponent)
            } catch {
                rejected.append(url.lastPathComponent)
            }
        }
        refresh()
        if !rejected.isEmpty {
            importMessage = "Not rvc-mlx models: \(rejected.joined(separator: ", ")). Convert them first (see the tutorial)."
        } else if !imported.isEmpty {
            importMessage = "Imported \(imported.count) file\(imported.count == 1 ? "" : "s")."
        }
    }

    func delete(_ voice: VoiceItem) {
        try? FileManager.default.removeItem(at: voice.url)
        refresh()
    }
}
