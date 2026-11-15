import PennyDesign
import XCTest

@MainActor
final class ChatLinkedDigestUITests: XCTestCase {
    func testDigestShowsEveryEntryAndEachLinkedHostOncePerEntry() {
        let app = XCUIApplication()
        app.launchStory("chatLinkedDigest")

        let row = app.element(AXID.chatMessageRow(104))
        let body = row.staticTexts.containing(NSPredicate(format: "label BEGINSWITH %@", "Here are a few new exhibits")).firstMatch
        XCTAssertTrue(body.waitForExistence(timeout: UITestWait.standard))
        let text = body.label

        XCTAssertTrue(text.contains("Coastal landscapes at Harbour Gallery — gallery.example.com"))
        XCTAssertTrue(text.contains("Night skies at City Hall — gallery.example.com"))
        XCTAssertTrue(text.contains("Winter gardens at Botanical Centre — www.city-botanical-gardens.example"))
        XCTAssertTrue(text.contains("Historic maps are on display, with more on the way."))
        XCTAssertFalse(text.contains("{link "))
        XCTAssertFalse(text.contains("www.archive.example"))
        XCTAssertEqual(text.components(separatedBy: "Some source links were unavailable.").count - 1, 1)
        XCTAssertTrue(text.contains("Let me know which one catches your eye."))
        XCTAssertEqual(text.components(separatedBy: "gallery.example.com").count - 1, 2)
        XCTAssertEqual(text.components(separatedBy: "www.city-botanical-gardens.example").count - 1, 1)

        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = "linked-digest"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
