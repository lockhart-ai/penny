import PennyDesign
import XCTest

/// Drives the app shell's real navigation (#3111): one stock
/// `NavigationStack`, a leading glass `Menu`/`Picker` page switcher, and a
/// system back button on a push. The old hand-built tab capsule shipped
/// with dead taps (#3105) because nothing exercised the real wiring — this
/// suite is the regression guard for that bug class, against the stock
/// elements that replaced it.
final class ShellNavigationUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchStory("uiTestChat")
    }

    /// Every page in the menu is reachable and shows its own root.
    func testEachPageReachableThroughTheMenu() {
        for tab in TopTab.allCases {
            let root = app.element(AXID.tabRoot(tab))
            app.button(AXID.pageMenu).tap()
            app.buttons[tab.name].tap()
            XCTAssertTrue(root.waitForExistence(timeout: UITestWait.standard),
                           "Expected \(tab)'s root to appear after choosing it from the page menu")
        }
    }

    /// Tapping a memory row pushes its detail with the system back button;
    /// tapping back returns to the list with the same row still there.
    func testMemoryRowPushesDetailAndSystemBackReturnsToList() {
        app.button(AXID.pageMenu).tap()
        app.buttons[TopTab.memory.name].tap()
        XCTAssertTrue(app.element(AXID.tabRoot(.memory)).waitForExistence(timeout: UITestWait.standard))

        let row = app.button(AXID.memoryRow("price-watch"))
        XCTAssertTrue(row.waitForExistence(timeout: UITestWait.standard), "Expected the fixture's price-watch row to load")

        let detailRoot = app.element(AXID.memoryDetailRoot)
        row.tap()
        XCTAssertTrue(detailRoot.waitForExistence(timeout: UITestWait.standard), "Expected the memory detail page to be pushed")

        let rowAgain = app.button(AXID.memoryRow("price-watch"))
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(rowAgain.waitForExistence(timeout: UITestWait.standard), "Expected the system back button to return to the list")
        XCTAssertFalse(detailRoot.exists, "Expected the detail page gone after the system back button")
    }

    /// A page switch never carries a stale push from the page it left: push
    /// Memory's detail, back out, switch to Settings and back to Memory —
    /// Memory shows its list, not the detail left pushed.
    func testSwitchingPagesAfterAPushedPageShowsEachPagesFreshRoot() {
        app.button(AXID.pageMenu).tap()
        app.buttons[TopTab.memory.name].tap()
        XCTAssertTrue(app.element(AXID.tabRoot(.memory)).waitForExistence(timeout: UITestWait.standard))

        let row = app.button(AXID.memoryRow("price-watch"))
        XCTAssertTrue(row.waitForExistence(timeout: UITestWait.standard))
        row.tap()
        let detailRoot = app.element(AXID.memoryDetailRoot)
        XCTAssertTrue(detailRoot.waitForExistence(timeout: UITestWait.standard))
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(row.waitForExistence(timeout: UITestWait.standard))

        app.button(AXID.pageMenu).tap()
        app.buttons[TopTab.settings.name].tap()
        XCTAssertTrue(app.element(AXID.tabRoot(.settings)).waitForExistence(timeout: UITestWait.standard),
                       "Expected Settings' own root")

        app.button(AXID.pageMenu).tap()
        app.buttons[TopTab.memory.name].tap()
        XCTAssertTrue(row.waitForExistence(timeout: UITestWait.standard), "Expected the memories list, not the earlier detail")
        XCTAssertFalse(detailRoot.exists, "Expected no leftover push after switching away and back")
    }

    /// While a collector runs, the page menu's toolbar item carries a stock
    /// badge with the live count.
    func testPageMenuShowsTheRunningCollectorBadge() {
        app.terminate()
        app.launchStory("chatCollectorsBadge")

        let menu = app.button(AXID.pageMenu)
        XCTAssertTrue(menu.waitForExistence(timeout: UITestWait.standard))
        XCTAssertEqual(menu.value as? String, "2",
                        "Expected the page menu's badge to show the fixture's running-collector count")
    }
}
