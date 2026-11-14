// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "PennyDesign",
    platforms: [.iOS(.v26)],
    products: [
        .library(
            name: "PennyDesign",
            targets: ["PennyDesign"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/SimplyDanny/SwiftLintPlugins.git", exact: "0.59.1")
    ],
    targets: [
        .target(
            name: "PennyDesign",
            plugins: [
                .plugin(name: "SwiftLintBuildToolPlugin", package: "SwiftLintPlugins")
            ]
        )
    ]
)
