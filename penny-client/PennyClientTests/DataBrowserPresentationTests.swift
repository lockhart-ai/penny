import Foundation
import SwiftUI
import Testing
import UIKit
@testable import PennyClient

@Suite(.serialized)
@MainActor
struct DataBrowserPresentationTests {
    @Test func syntheticBrowserLayoutsRenderAtPhoneAndPopoverSizes() async throws {
        let scenarios: [(String, CGSize, DataBrowserViewModel, ColorScheme, DynamicTypeSize)] = [
            ("phone-memories", CGSize(width: 393, height: 852), DataBrowserFixtures.model(), .light, .large),
            ("popover-activity", CGSize(width: 600, height: 720), DataBrowserFixtures.model(selection: "price-watch", panel: .activity), .dark, .large),
            ("phone-large-text", CGSize(width: 393, height: 852), DataBrowserFixtures.model(selection: "messages"), .light, .accessibility3),
            ("popover-prompts", CGSize(width: 600, height: 720), DataBrowserFixtures.model(tab: .prompts), .light, .large)
        ]
        for (name, size, model, scheme, textSize) in scenarios {
            model.updateConnection(true)
            model.refresh()
            let deadline = ContinuousClock.now.advanced(by: .seconds(3))
            while !model.loading.isEmpty {
                guard ContinuousClock.now < deadline else { Issue.record("Preview did not load"); return }
                await Task.yield()
            }
            #expect(model.errors.isEmpty)
            let content = DataBrowserContent(model: model, reconnect: {})
                .environment(\.colorScheme, scheme)
                .environment(\.dynamicTypeSize, textSize)
            let host = UIHostingController(rootView: content)
            let scene = try #require(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
            let previousKeyWindow = scene.keyWindow
            let window = UIWindow(windowScene: scene)
            defer {
                window.isHidden = true
                previousKeyWindow?.makeKeyAndVisible()
                model.stop()
            }
            window.frame = CGRect(origin: .zero, size: size)
            window.rootViewController = host
            window.makeKeyAndVisible()
            host.view.frame = window.bounds
            host.view.setNeedsLayout()
            host.view.layoutIfNeeded()
            await Task.yield()
            let image = UIGraphicsImageRenderer(size: size).image { _ in
                host.view.drawHierarchy(in: host.view.bounds, afterScreenUpdates: true)
            }
            let data = try #require(image.pngData())
            #expect(image.size == size)
            try data.write(to: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("browser-\(name).png"))
        }
    }
}
