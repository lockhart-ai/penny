import PennyDesign
import XCTest

/// The Memory and Prompts pages' search (#3150): both share the shell's
/// trailing `DefaultToolbarItem(kind: .search)` icon with Chat, and both
/// needed the same missing `.searchToolbarBehavior(.minimize)` to collapse
/// to a tappable icon instead of blanking the bar when tapped.
final class DataBrowserSearchUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchStory("uiTestChat")
    }

    /// The fixture's memories list (`price-watch`, `notes`, …) starts with
    /// no search field. Tapping the trailing icon opens it; typing "price"
    /// narrows the list to the one matching row — a real result change,
    /// not a pixel check. Closing removes the field again.
    func testMemorySearchOpensFiltersAndCloses() {
        app.button(AXID.pageMenu).tap()
        app.buttons[TopTab.memory.name].tap()
        XCTAssertTrue(app.element(AXID.tabRoot(.memory)).waitForExistence(timeout: UITestWait.standard))

        let priceRow = app.button(AXID.memoryRow("price-watch"))
        let notesRow = app.button(AXID.memoryRow("notes"))
        XCTAssertTrue(priceRow.waitForExistence(timeout: UITestWait.standard))
        XCTAssertTrue(notesRow.waitForExistence(timeout: UITestWait.standard))

        let field = app.searchFields.firstMatch
        XCTAssertFalse(field.exists, "Expected no search field before the trailing search icon is tapped")

        let searchButton = app.buttons["Search"]
        XCTAssertTrue(searchButton.waitForExistence(timeout: UITestWait.standard))
        searchButton.tap()

        XCTAssertTrue(field.waitForExistence(timeout: UITestWait.standard),
                       "Expected the search field to expand once the trailing icon is tapped")
        field.typeText("price")

        XCTAssertTrue(waitForAbsence(notesRow), "Expected 'notes' filtered out once 'price' narrowed the list")
        XCTAssertTrue(priceRow.exists, "Expected 'price-watch' to remain, since it matches the typed query")

        app.buttons["close"].tap()
        XCTAssertTrue(waitForAbsence(field), "Expected the search field gone after Close collapses it back to the icon")
    }

    /// The fixture's Prompts tab starts with both a chat run and several
    /// collector runs visible, no search field. Tapping the trailing icon
    /// opens it; typing "chat" narrows the list to the chat run alone — a
    /// real result change. Closing removes the field again.
    func testPromptsSearchOpensFiltersAndCloses() {
        app.button(AXID.pageMenu).tap()
        app.buttons[TopTab.prompts.name].tap()
        XCTAssertTrue(app.element(AXID.tabRoot(.prompts)).waitForExistence(timeout: UITestWait.standard))

        let chatRun = app.staticTexts["chat"]
        let collectorRun = app.staticTexts["collector"]
        XCTAssertTrue(chatRun.waitForExistence(timeout: UITestWait.standard))
        XCTAssertTrue(collectorRun.waitForExistence(timeout: UITestWait.standard))

        let field = app.searchFields.firstMatch
        XCTAssertFalse(field.exists, "Expected no search field before the trailing search icon is tapped")

        let searchButton = app.buttons["Search"]
        XCTAssertTrue(searchButton.waitForExistence(timeout: UITestWait.standard))
        searchButton.tap()

        XCTAssertTrue(field.waitForExistence(timeout: UITestWait.standard),
                       "Expected the search field to expand once the trailing icon is tapped")
        field.typeText("chat")

        XCTAssertTrue(waitForAbsence(collectorRun), "Expected the collector runs filtered out once 'chat' narrowed the list")
        XCTAssertTrue(chatRun.exists, "Expected the chat run to remain, since it matches the typed query")

        app.buttons["close"].tap()
        XCTAssertTrue(waitForAbsence(field), "Expected the search field gone after Close collapses it back to the icon")
    }
}
