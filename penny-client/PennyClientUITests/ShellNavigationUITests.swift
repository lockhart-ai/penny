import PennyDesign
import XCTest

/// Drives the app shell's real navigation: the top capsule's tabs and a
/// pushed page's back control. The tab bar shipped with dead taps once
/// (#3105) because nothing exercised the real `onSelect` wiring.
///
/// Building this suite found a live instance of the same bug class, still
/// on `main` — see #3111 for the reproduction and root-cause work. The
/// `XCTExpectFailure` calls below land the finding without leaving the gate
/// red; a follow-up fix PR removes them.
final class ShellNavigationUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
    }

    /// Every tab in the top capsule is tappable and shows its own screen.
    /// Chat and Memory are a real regression guard for #3105. Prompts and
    /// Settings reproduce #3111 instead.
    func testTappingEachTabShowsItsScreen() {
        app.launchStory("uiTestChat")

        for tab in TopTab.allCases {
            let root = app.element(AXID.tabRoot(tab))
            app.button(AXID.tabButton(tab)).tap()
            let shown = root.waitForExistence(timeout: UITestWait.standard)

            switch tab {
            case .chat, .memory:
                XCTAssertTrue(shown, "Expected \(tab)'s root to appear after tapping its tab button")
            case .prompts, .settings:
                XCTExpectFailure("#3111: tapping the \(tab) tab button never reaches TopTabBar's onSelect") {
                    XCTAssertTrue(shown)
                }
            }
        }
    }

    /// Tapping a memory row pushes its detail; the back circle is meant to
    /// return to the list with the same row still there — it reproduces
    /// #3111 instead (see the type doc).
    func testMemoryRowPushesDetailAndBackReturnsToList() {
        app.launchStory("uiTestChat")

        let memoryRoot = app.element(AXID.tabRoot(.memory))
        app.button(AXID.tabButton(.memory)).tap()
        XCTAssertTrue(memoryRoot.waitForExistence(timeout: UITestWait.standard))

        let row = app.button(AXID.memoryRow("price-watch"))
        XCTAssertTrue(row.waitForExistence(timeout: UITestWait.standard), "Expected the fixture's price-watch row to load")

        let detailRoot = app.element(AXID.memoryDetailRoot)
        row.tap()
        XCTAssertTrue(detailRoot.waitForExistence(timeout: UITestWait.standard), "Expected the memory detail page to be pushed")

        let rowAgain = app.button(AXID.memoryRow("price-watch"))
        app.button(AXID.backButton).tap()
        let backWorked = rowAgain.waitForExistence(timeout: UITestWait.standard)
        XCTExpectFailure("#3111: tapping the back circle never reaches AppShellView's goBack()") {
            XCTAssertTrue(backWorked)
        }
    }
}
