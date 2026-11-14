import PennyDesign
import XCTest

/// Exercises the composer and chat scrolling with local story fixtures.
final class ComposerUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
    }

    func testSendAppearsWithTextAndSendingAddsMessageAndClearsTheField() {
        app.launchStory("uiTestChat")
        let sendButton = app.button(AXID.composerSendButton)
        XCTAssertTrue(waitForAbsence(sendButton), "Expected no send button with an empty draft")

        let field = app.textField(AXID.composerTextField)
        XCTAssertTrue(field.waitForExistence(timeout: UITestWait.standard))
        field.tap()
        let draft = "A UI test's composer message, sent to prove it lands"
        field.typeText(draft)

        XCTAssertTrue(waitForVisibility(sendButton, above: app.keyboards.firstMatch),
                      "Expected the send button above the keyboard once there's a draft")

        let sentText = app.staticTexts[draft]
        sendButton.tap()
        XCTAssertTrue(sentText.waitForExistence(timeout: UITestWait.standard), "Expected the sent text to appear as a new message")
        // `Composer.hasDraft` is `!text.isEmpty || pendingImage != nil`, so
        // the send button only shows with a draft — its disappearance here
        // is the structural proof the field went back to empty, more
        // reliable than reading a text field's accessibility value (which
        // can echo the placeholder instead of "" on an empty field).
        XCTAssertTrue(waitForAbsence(sendButton), "Expected the send button to disappear once the draft was cleared")

        capture(named: "composer-after-send")
    }

    func testLatestMessageMovesAboveComposerWhenKeyboardOpens() {
        app.launchStory("uiTestChatScrolling")
        let field = app.textField(AXID.composerTextField)
        let latest = app.staticTexts["This is the latest message."]
        XCTAssertTrue(waitForVisibility(latest, above: field))
        let initialY = latest.frame.maxY

        field.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: UITestWait.standard))
        XCTAssertTrue(waitForVisibility(latest, above: field))
        XCTAssertLessThan(latest.frame.maxY, initialY)
        capture(named: "composer-keyboard-open")
    }

    func testDelayedStatusWidgetIsVisibleAfterSending() {
        app.launchStory("uiTestChatScrolling")
        let dock = app.element(AXID.liveDock)
        let field = app.textField(AXID.composerTextField)
        XCTAssertTrue(field.waitForExistence(timeout: UITestWait.standard))
        XCTAssertFalse(dock.exists)
        field.tap()
        field.typeText("Please check the latest request.")
        let sendButton = app.button(AXID.composerSendButton)
        XCTAssertTrue(waitForVisibility(sendButton, above: app.keyboards.firstMatch))
        sendButton.tap()

        XCTAssertTrue(waitForVisibility(dock, above: field), "Expected the whole status widget above the composer")
        capture(named: "composer-status-after-send")
    }

    private func waitForVisibility(_ element: XCUIElement, above reference: XCUIElement) -> Bool {
        let predicate = NSPredicate { _, _ in
            element.exists && element.isHittable && reference.exists && element.frame.maxY <= reference.frame.minY
        }
        let expectation = XCTNSPredicateExpectation(predicate: predicate, object: nil)
        return XCTWaiter().wait(for: [expectation], timeout: UITestWait.standard) == .completed
    }

    private func capture(named name: String) {
        let screenshot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        screenshot.name = name
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }
}
