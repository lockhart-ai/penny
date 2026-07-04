// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import CompilerPluginSupport

let package = Package(
    name: "SQLPropertyMacros",
    platforms: [.macOS(.v10_15), .iOS(.v13), .tvOS(.v13), .watchOS(.v6), .macCatalyst(.v13)],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "SQLPropertyMacros",
            targets: ["SQLPropertyMacros"]
        ),
        .executable(
            name: "SQLPropertyMacrosClient",
            targets: ["SQLPropertyMacrosClient"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-syntax.git", from: "601.0.0-latest"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .macro(
            name: "SQLPropertyMacrosImplementation",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ]
        ),
        .target(name: "SQLPropertyMacros", dependencies: ["SQLPropertyMacrosImplementation"]),
        .executableTarget(
            name: "SQLPropertyMacrosClient",
            dependencies: ["SQLPropertyMacros"]
        ),
        .testTarget(
            name: "SQLPropertyMacrosTests",
            dependencies: [
                "SQLPropertyMacros",
                .product(name: "SwiftSyntaxMacrosTestSupport", package: "swift-syntax"),
            ]
        ),
    ]
)
