import PennyDesign
import XCTest

/// Drives the app shell's real navigation (#3106, #3105): the top capsule's
/// tabs and a pushed page's back control. The tab bar shipped with dead
/// taps once (#3105) because nothing exercised the real `onSelect` wiring.
///
/// Building this suite found a live instance of exactly that class of bug,
/// still present on `main`: tapping the Prompts or Settings tab button, or
/// the back circle, never reaches its `onTap`/`onSelect` closure — confirmed
/// with `NSLog` instrumentation inside `AppShellView` (removed again), which
/// showed the closure body never executing, independent of tap order, a
/// retry loop, an `isHittable` wait, or `press(forDuration:)` in place of
/// `tap()`. Chat and Memory (the two leftmost tab buttons) are unaffected.
/// The `XCTExpectFailure` calls below keep this reproduction landing instead
/// of leaving the gate red; remove each one (and tighten the assertion back
/// to a plain `XCTAssertTrue`) once its control is fixed — `XCTExpectFailure`
/// itself fails the suite the day that happens, so nothing has to remember
/// to come back and check.
final class ShellNavigationUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
    }

    /// Every tab in the top capsule is tappable and shows its own screen.
    /// Chat and Memory are a real regression guard for #3105. Prompts and
    /// Settings reproduce a live, unfixed instance of the same bug class.
    func testTappingEachTabShowsItsScreen() {
        app.launchStory("uiTestChat")

        for tab in TopTab.allCases {
            let root = app.element(AXID.tabRoot(tab))
            let button = app.button(AXID.tabButton(tab))
            switch tab {
            case .chat, .memory:
                XCTAssertTrue(
                    tap(button, until: root.exists, timeout: UITestWait.standard),
                    "Expected \(tab)'s root to appear after tapping its tab button"
                )
            case .prompts, .settings:
                XCTExpectFailure("Tapping the \(tab) tab button never reaches TopTabBar's onSelect — live bug, not a test gap") {
                    XCTAssertTrue(tap(button, until: root.exists, timeout: UITestWait.standard))
                }
            }
        }
    }

    /// Tapping a memory row pushes its detail. The back circle is meant to
    /// return to the list with the same row still there — it reproduces the
    /// dead-tap bug above instead (see the type doc).
    func testMemoryRowPushesDetailAndBackReturnsToList() {
        app.launchStory("uiTestChat")

        let memoryRoot = app.element(AXID.tabRoot(.memory))
        XCTAssertTrue(tap(app.button(AXID.tabButton(.memory)), until: memoryRoot.exists, timeout: UITestWait.standard))

        let row = app.button(AXID.memoryRow("price-watch"))
        XCTAssertTrue(row.waitForExistence(timeout: UITestWait.standard), "Expected the fixture's price-watch row to load")

        let detailRoot = app.element(AXID.memoryDetailRoot)
        XCTAssertTrue(
            tap(row, until: detailRoot.exists, timeout: UITestWait.standard),
            "Expected the memory detail page to be pushed"
        )

        let rowAgain = app.button(AXID.memoryRow("price-watch"))
        XCTExpectFailure("Tapping the back circle never reaches AppShellView's goBack() — live bug, not a test gap") {
            XCTAssertTrue(tap(app.button(AXID.backButton), until: rowAgain.exists, timeout: UITestWait.standard))
        }
    }
}
