import RVCKit
import SwiftUI

@main
struct RVCPocketApp: App {
    @StateObject private var library = ModelLibrary()
    @StateObject private var converter = Converter()

    init() {
        RVCRuntime.setGPUCacheLimit(megabytes: 256)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(library)
                .environmentObject(converter)
                .tint(.accentColor)
                .onOpenURL { url in library.importFiles([url]) }  // AirDrop / "Open in RVC Pocket"
        }
    }
}
