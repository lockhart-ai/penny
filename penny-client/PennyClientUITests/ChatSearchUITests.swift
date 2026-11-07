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
    /// opening search hides the composer immediately, even before anything
    /// is typed (#3150 — the approved mock shows no composer during search
    /// at all, and two stacked text fields read badly). Typing a fixture
    /// word highlights every match's bubble and marks the newest one as the
    /// current match; the system Close control clears the field, brings
    /// the composer back, and collapses the field to the icon.
    ///
    /// Highlight state is read off `AXID.chatMessageRow`'s
    /// `.accessibilityValue` (#3150) — a non-pixel, non-existence check.
    /// A weaker version of this test once asserted only that a matched
    /// message's own text was visible on screen, which is true with or
    /// without a search (the fixture's messages are already on screen
    /// before any search starts), and that blind spot let the highlight
    /// silently never apply at all (`ChatSearchState.highlightQuery`'s
    /// `isSearching` gate never opened — a `@Environment(\.isSearching)`
    /// read directly on `MessageView` can't see what `.searchable`,
    /// applied to that same view's own output, sets for its subtree; see
    /// `SearchActivationBridge` in `MessageView.swift`).
    func testSearchOpensFromTheTrailingIconHighlightsEveryMatchAndCloseCloses() {
        let field = app.searchFields.firstMatch
        XCTAssertFalse(field.exists, "Expected no search field before the trailing search icon is tapped")
        XCTAssertTrue(app.textFields[AXID.composerTextField].waitForExistence(timeout: UITestWait.standard),
                       "Expected the composer visible before search opens")

        let searchButton = app.buttons["Search"]
        XCTAssertTrue(searchButton.waitForExistence(timeout: UITestWait.standard))
        searchButton.tap()

        XCTAssertTrue(field.waitForExistence(timeout: UITestWait.standard),
                       "Expected the search field to expand once the trailing icon is tapped")
        XCTAssertTrue(waitForAbsence(app.textFields[AXID.composerTextField]),
                       "Expected the composer hidden as soon as search opens, before anything is typed")

        field.typeText("ferry")
        XCTAssertEqual(field.value as? String, "ferry")

        // All three `uiTestChat` fixture messages (ids 101-103) contain
        // "ferry"; the newest (103, "Yes, please book the ferry.") is the
        // current match `ChatSearchState` surfaces and scrolls to.
        let currentMatch = app.element(AXID.chatMessageRow(103))
        let olderMatch = app.element(AXID.chatMessageRow(102))
        XCTAssertTrue(currentMatch.waitForExistence(timeout: UITestWait.standard))
        XCTAssertTrue(olderMatch.waitForExistence(timeout: UITestWait.standard))
        XCTAssertTrue(waitForValue(currentMatch, toEqual: "highlighted-current"),
                       "Expected the newest match's bubble marked as the current highlight")
        XCTAssertTrue(waitForValue(olderMatch, toEqual: "highlighted"),
                       "Expected an older match's bubble highlighted, not marked current")

        app.buttons["close"].tap()
        XCTAssertTrue(waitForAbsence(field), "Expected the search field gone after Close collapses it back to the icon")
        XCTAssertTrue(app.textFields[AXID.composerTextField].waitForExistence(timeout: UITestWait.standard),
                       "Expected the composer back once search closes")
    }
}
