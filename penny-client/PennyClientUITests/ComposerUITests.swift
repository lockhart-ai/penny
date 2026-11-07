import PennyDesign
import XCTest

/// The composer (#3106): the send arrow only appears once there's a draft,
/// and sending adds the message and clears the draft — against the
/// `uiTestChat` fixture's connected, server-free chat.
final class ComposerUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchStory("uiTestChat")
    }

    func testSendAppearsWithTextAndSendingAddsMessageAndClearsTheField() {
        let sendButton = app.button(AXID.composerSendButton)
        XCTAssertTrue(waitForAbsence(sendButton), "Expected no send button with an empty draft")

        let field = app.textField(AXID.composerTextField)
        XCTAssertTrue(field.waitForExistence(timeout: UITestWait.standard))
        field.tap()
        let draft = "A UI test's composer message, sent to prove it lands"
        field.typeText(draft)

        XCTAssertTrue(sendButton.waitForExistence(timeout: UITestWait.standard), "Expected the send button once there's a draft")

        let sentText = app.staticTexts[draft]
        sendButton.tap()
        XCTAssertTrue(sentText.waitForExistence(timeout: UITestWait.standard), "Expected the sent text to appear as a new message")
        // `Composer.hasDraft` is `!text.isEmpty || pendingImage != nil`, so
        // the send button only shows with a draft — its disappearance here
        // is the structural proof the field went back to empty, more
        // reliable than reading a text field's accessibility value (which
        // can echo the placeholder instead of "" on an empty field).
        XCTAssertTrue(waitForAbsence(sendButton), "Expected the send button to disappear once the draft was cleared")
    }
}
