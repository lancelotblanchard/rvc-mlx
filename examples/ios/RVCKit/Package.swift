// swift-tools-version: 5.9
// RVCKit: RVC voice conversion on MLX for iOS and macOS (a Swift port of the rvc-mlx engine).
import PackageDescription

let package = Package(
    name: "RVCKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "RVCKit", targets: ["RVCKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.0"),
    ],
    targets: [
        .target(
            name: "RVCKit",
            dependencies: [.product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "RVCKitTests",
            dependencies: ["RVCKit"]
        ),
    ]
)
