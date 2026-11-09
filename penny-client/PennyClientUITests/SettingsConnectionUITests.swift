import PennyDesign
import XCTest

/// Regression guard for #3166 — a real app launch (no `-PennyStory` fixture),
/// because the bug was in the live Settings/Prefs path, not anything a story
/// stubs out. The Connection section's bottom-bar Save (#3140) sits exactly
/// where the keyboard covers it while a field has focus, so the code owner
/// could type a WebSocket URL but never reach Save to persist it. The fix
/// saves on keyboard submit (the return key) instead, so this drives the URL
/// field, presses return, and asserts the value survived a relaunch.
final class SettingsConnectionUITests: XCTestCase {
    func testSubmittingTheWebSocketURLPersistsAcrossRelaunch() {
        continueAfterFailure = false
        var app = XCUIApplication()
        app.launch()

        app.button(AXID.pageMenu).tap()
        app.buttons[TopTab.settings.name].tap()
        XCTAssertTrue(app.element(AXID.tabRoot(.settings)).waitForExistence(timeout: UITestWait.standard))

        let urlField = app.textFields["WebSocket URL"]
        XCTAssertTrue(urlField.waitForExistence(timeout: UITestWait.standard), "Expected the WebSocket URL field")
        urlField.tap()
        urlField.typeText("ws://192.168.1.50:9091\n")

        app.terminate()
        app = XCUIApplication()
        app.launch()

        app.button(AXID.pageMenu).tap()
        app.buttons[TopTab.settings.name].tap()
        let urlFieldAfterRelaunch = app.textFields["WebSocket URL"]
        XCTAssertTrue(urlFieldAfterRelaunch.waitForExistence(timeout: UITestWait.standard))
        XCTAssertEqual(urlFieldAfterRelaunch.value as? String, "ws://192.168.1.50:9091",
                        "Expected the saved URL to persist across relaunch")
    }
}
