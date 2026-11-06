import Foundation
import SwiftUI
import Testing
import UIKit
import PennyServices
@testable import PennyClient

@Suite(.serialized)
@MainActor
struct DataBrowserPresentationTests {
    /// One hosted-render scenario: the tab's root, its fixtures model, and
    /// the window it renders into.
    private struct Scenario {
        let name: String
        let size: CGSize
        let model: DataBrowserViewModel
        let tab: DataBrowserViewModel.Tab
        let scheme: ColorScheme
        let textSize: DynamicTypeSize
    }

    /// The browser tabs render under the app shell's tab roots (#3037) —
    /// phone sizes, plus one wide size for the iPad hosting.
    @Test func syntheticBrowserLayoutsRenderAtHostedSizes() async throws {
        let phone = CGSize(width: 393, height: 852)
        let wide = CGSize(width: 768, height: 1024)
        let scenarios: [Scenario] = [
            Scenario(name: "memories", size: phone, model: DataBrowserFixtures.model(), tab: .memories, scheme: .light, textSize: .large),
            Scenario(name: "detail-activity", size: phone, model: DataBrowserFixtures.model(selection: "price-watch", panel: .activity), tab: .memories, scheme: .dark, textSize: .large),
            Scenario(name: "memories-large-text", size: phone, model: DataBrowserFixtures.model(selection: "messages"), tab: .memories, scheme: .light, textSize: .accessibility3),
            Scenario(name: "prompts", size: phone, model: DataBrowserFixtures.model(tab: .prompts), tab: .prompts, scheme: .light, textSize: .large),
            Scenario(name: "linked-run", size: phone, model: DataBrowserViewModel(runID: "preview-worked", read: DataBrowserFixtures.read), tab: .prompts, scheme: .light, textSize: .large),
            Scenario(name: "missing-run", size: phone, model: DataBrowserViewModel(runID: "missing-run", read: DataBrowserFixtures.read), tab: .prompts, scheme: .dark, textSize: .accessibility3),
            Scenario(name: "memories-wide", size: wide, model: DataBrowserFixtures.model(), tab: .memories, scheme: .light, textSize: .large)
        ]
        for scenario in scenarios {
            let name = scenario.name
            let size = scenario.size
            let model = scenario.model
            let tab = scenario.tab
            let scheme = scenario.scheme
            let textSize = scenario.textSize
            model.updateConnection(true)
            model.refresh()
            let deadline = ContinuousClock.now.advanced(by: .seconds(3))
            while !model.loading.isEmpty {
                guard ContinuousClock.now < deadline else { Issue.record("Preview did not load"); return }
                await Task.yield()
            }
            #expect(model.errors.isEmpty)
            let content = Group {
                switch tab {
                case .memories: DataBrowserMemoriesView(model: model, reconnect: {})
                case .prompts: DataBrowserPromptsView(model: model, reconnect: {})
                }
            }
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
