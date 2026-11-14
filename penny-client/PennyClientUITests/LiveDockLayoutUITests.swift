import PennyDesign
import XCTest

@MainActor
final class LiveDockLayoutUITests: XCTestCase {
    func testWorkingDockAdaptsToRotation() {
        continueAfterFailure = false
        XCUIDevice.shared.orientation = .portrait
        defer { XCUIDevice.shared.orientation = .portrait }
        let app = XCUIApplication()
        app.launchStory("chatWorkingLong")

        let dock = app.element(AXID.liveDock)
        let stream = dock.scrollViews.firstMatch
        let composer = app.textField(AXID.composerTextField)
        XCTAssertTrue(dock.waitForExistence(timeout: UITestWait.standard))
        XCTAssertTrue(composer.waitForExistence(timeout: UITestWait.standard))
        XCTAssertTrue(composer.isHittable)
        XCTAssertTrue(stream.waitForExistence(timeout: UITestWait.standard))
        let expandedHeight = stream.frame.height
        XCTAssertGreaterThan(expandedHeight, 0)
        capture(named: "working-dock-portrait")

        XCUIDevice.shared.orientation = .landscapeLeft
        app.swipeUp()
        for _ in 0..<8 where !dock.isHittable {
            app.scrollViews.firstMatch.swipeUp()
        }
        XCTAssertTrue(dock.isHittable)
        let shrunk = XCTNSPredicateExpectation(
            predicate: NSPredicate { _, _ in
                app.frame.width > app.frame.height && stream.frame.height < expandedHeight - 10
            },
            object: nil
        )
        XCTAssertEqual(XCTWaiter().wait(for: [shrunk], timeout: UITestWait.standard), .completed,
                       "Expected stream to shrink from \(expandedHeight); got \(stream.frame.height)")
        XCTAssertTrue(composer.isHittable)
        capture(named: "working-dock-landscape")
    }

    private func capture(named name: String) {
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
