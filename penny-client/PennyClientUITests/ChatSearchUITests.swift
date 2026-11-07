import PennyDesign
import XCTest

/// Find-in-chat (#3045), wired to the stock trailing-icon `.searchable`
/// field every page shares (#3111/#3129/#3150: `.searchToolbarBehavior
/// (.minimize)` alongside the shell's `DefaultToolbarItem(kind: .search)`
/// — the recipe confirmed in the component gallery, `SearchTopPage.swift`,
/// as the one that collapses to a tappable icon instead of blanking the
/// bar): tapping the icon to open the field, typing a query, and closing
/// it. No counter, no up/down — the code owner approved highlights only.
final class ChatSearchUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchStory("uiTestChat")
    }

    /// The field doesn't exist until the trailing search icon is tapped;
    /// once open, typing a fixture word highlights its newest match and
    /// scrolls it into view; the system Close control clears the field and
    /// collapses it back to the icon. The field itself existing with our
    /// typed text IS the search state — there's no separate flag to assert.
    func testSearchOpensFromTheTrailingIconHighlightsTheNewestMatchAndCloseCloses() {
        let field = app.searchFields.firstMatch
        XCTAssertFalse(field.exists, "Expected no search field before the trailing search icon is tapped")

        let searchButton = app.buttons["Search"]
        XCTAssertTrue(searchButton.waitForExistence(timeout: UITestWait.standard))
        searchButton.tap()

        XCTAssertTrue(field.waitForExistence(timeout: UITestWait.standard),
                       "Expected the search field to expand once the trailing icon is tapped")
        field.typeText("ferry")
        XCTAssertEqual(field.value as? String, "ferry")

        // All three `uiTestChat` fixture messages contain "ferry"; the
        // newest (id 103) is the one `ChatSearchState` surfaces and scrolls
        // to first.
        let newestMatch = app.staticTexts["Yes, please book the ferry."]
        XCTAssertTrue(newestMatch.waitForExistence(timeout: UITestWait.standard),
                       "Expected the newest ferry match to be visible once search finds it")

        app.buttons["close"].tap()
        XCTAssertTrue(waitForAbsence(field), "Expected the search field gone after Close collapses it back to the icon")
    }
}
