// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "PennyServices",
    platforms: [.iOS(.v18)],
    products: [
        .library(
            name: "PennyServices",
            targets: ["PennyServices"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/stephencelis/SQLite.swift.git", from: "0.16.0"),
        .package(path: "../SQLPropertyMacros"),
        .package(url: "https://github.com/realm/SwiftLint.git", from: "0.59.1"),
    ],
    targets: [
        .target(
            name: "PennyServices",
            dependencies: [
                .product(name: "SQLite", package: "SQLite.swift"),
                .product(name: "SQLPropertyMacros", package: "SQLPropertyMacros"),
            ],
            resources: [
                .process("MessageSearch.metal"),
            ],
            swiftSettings: [
                .unsafeFlags(["-strict-concurrency=minimal"]),
            ],
            plugins: [
                .plugin(name: "SwiftLintBuildToolPlugin", package: "SwiftLint"),
            ]
        ),
        .testTarget(
            name: "PennyServicesTests",
            dependencies: ["PennyServices"],
            swiftSettings: [
                .unsafeFlags(["-strict-concurrency=minimal"]),
            ],
            plugins: [
                .plugin(name: "SwiftLintBuildToolPlugin", package: "SwiftLint"),
            ]
        ),
    ]
)
