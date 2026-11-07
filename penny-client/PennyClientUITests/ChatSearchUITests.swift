import PennyDesign
import XCTest

/// Find-in-chat (#3045): opening the bar, typing a query, stepping between
/// matches, and closing it. Building this suite found that four of its
/// five controls reproduce the same live bug as `ShellNavigationUITests` —
/// see #3111 for the reproduction and root-cause work. The `XCTExpectFailure`
/// calls below land the finding without leaving the gate red; a follow-up
/// fix PR removes them.
final class ChatSearchUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
    }

    /// The search circle is meant to open find-in-chat — it reproduces
    /// #3111 instead.
    func testSearchCircleOpensTheBar() {
        app.launchStory("uiTestChat")
        let field = app.textField(AXID.searchField)
        app.button(AXID.searchOpenButton).tap()
        let opened = field.waitForExistence(timeout: UITestWait.standard)
        XCTExpectFailure("#3111: tapping the search circle never reaches AppShellView's openChatSearch()") {
            XCTAssertTrue(opened)
        }
    }

    /// With the bar already open (the `chatSearch` story opens it in code,
    /// never by a tap — independent of the bug above) and three known
    /// "ferry" matches already found, the counter reads exactly "1 of 3",
    /// a real passing assertion with no button involved. Stepping to the
    /// older match and closing the bar are meant to move/dismiss it — both
    /// reproduce #3111 instead.
    func testStepAndCloseFromAnAlreadyOpenBar() {
        app.launchStory("chatSearch")
        let counter = app.element(AXID.searchCounter)
        XCTAssertTrue(counter.waitForExistence(timeout: UITestWait.standard))
        XCTAssertEqual(counter.label, "1 of 3", "Expected all three fixture messages containing \"ferry\" to match")

        app.button(AXID.searchOlderButton).tap()
        let stepped = waitForLabel(counter, toEqual: "2 of 3")
        XCTExpectFailure("#3111: tapping the older-match circle never reaches ChatSearchBar's onStepToOlderMatch") {
            XCTAssertTrue(stepped, "Expected stepping to the older match to advance the counter by one")
        }

        let field = app.textField(AXID.searchField)
        app.button(AXID.searchCloseButton).tap()
        let closed = waitForAbsence(field)
        XCTExpectFailure("#3111: tapping the close circle never reaches ChatSearchBar's onClose") {
            XCTAssertTrue(closed, "Expected the search field to be dismissed after Close")
        }
    }
}
