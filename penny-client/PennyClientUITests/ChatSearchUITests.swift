import PennyDesign
import XCTest

/// Find-in-chat (#3045), wired to the stock `.searchable` field under
/// Chat's title (#3111/#3129, `.navigationBarDrawer(displayMode: .always)`
/// — confirmed the only placement on this SDK that doesn't sit the search
/// bar on top of the composer): typing a query, and cancelling it. No
/// counter, no up/down — the code owner approved highlights only.
final class ChatSearchUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchStory("uiTestChat")
    }

    /// Typing a fixture word highlights its newest match and scrolls it
    /// into view; the system Cancel control closes the field again. The
    /// field itself existing with our typed text IS the search state —
    /// there's no separate flag to assert.
    func testSearchHighlightsTheNewestMatchAndCancelCloses() {
        let field = app.searchFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: UITestWait.standard))

        field.tap()
        field.typeText("ferry")
        XCTAssertEqual(field.value as? String, "ferry")

        // All three `uiTestChat` fixture messages contain "ferry"; the
        // newest (id 103) is the one `ChatSearchState` surfaces and scrolls
        // to first.
        let newestMatch = app.staticTexts["Yes, please book the ferry."]
        XCTAssertTrue(newestMatch.waitForExistence(timeout: UITestWait.standard),
                       "Expected the newest ferry match to be visible once search finds it")

        app.buttons["Cancel"].tap()
        XCTAssertTrue(waitForAbsence(app.keyboards.firstMatch), "Expected Cancel to dismiss the keyboard")
        XCTAssertEqual(field.value as? String, "Search", "Expected Cancel to clear the query back to the placeholder")
    }
}
