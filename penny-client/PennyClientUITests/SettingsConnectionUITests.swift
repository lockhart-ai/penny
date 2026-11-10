import PennyDesign
import XCTest

/// Regression guard for #3166 and #3169 — a real app launch (no
/// `-PennyStory` fixture), because the Connection section's behavior under
/// test is the live Settings/Prefs path, not anything a story stubs out.
/// Two launch arguments keep this suite from disturbing the simulator's own
/// state: `-PennyUITestsSkipNotificationPrompt` skips the system
/// notification-permission alert `AppDelegate` would otherwise show on a
/// fresh simulator (a UI test is unattended and has nothing to dismiss it
/// with), and `-PennyUITestsIsolatedPrefs` writes the WebSocket URL this
/// suite types into its own `UserDefaults` suite instead of the real one,
/// so it can never leak into a later manual run of the app.
///
/// #3140's bottom-bar Save sat exactly where the keyboard covers it while a
/// field has focus, so the code owner could type a WebSocket URL but never
/// reach Save to persist it (#3166's bug). #3169 removes that bar entirely
/// — the Connection section now saves on keyboard submit (the return key)
/// and on the field losing focus with no Return pressed — so this suite
/// covers both triggers.
final class SettingsConnectionUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        launch(app)
    }

    func testSubmittingTheWebSocketURLPersistsAcrossRelaunch() {
        let urlField = openSettingsAndFindURLField()
        urlField.tap()
        urlField.typeText("ws://192.168.1.50:9091\n")

        let urlFieldAfterRelaunch = relaunchToSettingsURLField()
        XCTAssertEqual(urlFieldAfterRelaunch.value as? String, "ws://192.168.1.50:9091",
                        "Expected the saved URL to persist across relaunch")
    }

    /// #3169's other save trigger: losing focus with no Return pressed —
    /// tapping the Username field moves focus away from the URL field
    /// without ever submitting it.
    func testEditingTheWebSocketURLAndTappingElsewherePersistsAcrossRelaunch() {
        let urlField = openSettingsAndFindURLField()
        urlField.tap()
        urlField.typeText("ws://10.0.0.7:9091")

        let usernameField = app.textFields["Username"]
        XCTAssertTrue(usernameField.waitForExistence(timeout: UITestWait.standard), "Expected the Username field")
        usernameField.tap()

        let urlFieldAfterRelaunch = relaunchToSettingsURLField()
        XCTAssertEqual(urlFieldAfterRelaunch.value as? String, "ws://10.0.0.7:9091",
                        "Expected the saved URL to persist across relaunch even though Return was never pressed")
    }

    private func launch(_ app: XCUIApplication) {
        app.launchArguments = ["-PennyUITestsSkipNotificationPrompt", "-PennyUITestsIsolatedPrefs"]
        app.launch()
    }

    private func openSettingsAndFindURLField() -> XCUIElement {
        app.button(AXID.pageMenu).tap()
        app.buttons[TopTab.settings.name].tap()
        XCTAssertTrue(app.element(AXID.tabRoot(.settings)).waitForExistence(timeout: UITestWait.standard))

        let urlField = app.textFields["WebSocket URL"]
        XCTAssertTrue(urlField.waitForExistence(timeout: UITestWait.standard), "Expected the WebSocket URL field")
        return urlField
    }

    private func relaunchToSettingsURLField() -> XCUIElement {
        app.terminate()
        app = XCUIApplication()
        launch(app)

        app.button(AXID.pageMenu).tap()
        app.buttons[TopTab.settings.name].tap()
        let urlField = app.textFields["WebSocket URL"]
        XCTAssertTrue(urlField.waitForExistence(timeout: UITestWait.standard))
        return urlField
    }
}
