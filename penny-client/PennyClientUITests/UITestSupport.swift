import XCTest

/// Shared launch/lookup helpers for the whole `PennyClientUITests` smoke
/// suite (#3106) — every test launches against fixture data via
/// `-PennyStory <id>` (#2965), never a server, so the suite is deterministic
/// and needs no network.
extension XCUIApplication {
    /// Launches against one named story (`StoryID`'s raw value) instead of
    /// the live app.
    func launchStory(_ id: String) {
        launchArguments = ["-PennyStory", id]
        launch()
    }

    /// Looks a non-interactive view up by its accessibility identifier,
    /// regardless of the kind of element SwiftUI rendered it as (static
    /// text, or a plain container tagged for a UI test, like a tab root) —
    /// for existence/label checks only. A `descendants(matching: .any)`
    /// query can resolve to a stale, about-to-be-removed element during a
    /// SwiftUI transition when two elements briefly share an identifier;
    /// `button(_:)` below exists so every *tap* instead goes through
    /// XCUITest's own button-type matching, which resolves that ambiguity
    /// the way `.any` does not.
    func element(_ identifier: String) -> XCUIElement {
        descendants(matching: .any)[identifier]
    }

    /// Looks a button up by its accessibility identifier for tapping —
    /// every control this suite drives is a SwiftUI `Button`, so matching
    /// by `.buttons` (not `.any`) is both the idiomatic XCUITest lookup and
    /// the reliable one.
    func button(_ identifier: String) -> XCUIElement {
        buttons[identifier]
    }

    /// Looks a text field up by its accessibility identifier for typing —
    /// the same reasoning as `button(_:)`, matched by `.textFields`.
    func textField(_ identifier: String) -> XCUIElement {
        textFields[identifier]
    }
}

/// A generous, CI-safe wait: every test polls for the expected state with
/// `waitForExistence(timeout:)` instead of sleeping a fixed duration (the
/// "Hard-won patterns" rule, #3102).
enum UITestWait {
    static let standard: TimeInterval = 10
}

/// Polls `element`'s label for an exact value (e.g. the search counter's
/// "n of m" text) instead of sleeping — the same no-fixed-sleep rule
/// `waitForExistence` follows, extended to a value rather than only
/// existence.
@discardableResult
func waitForLabel(_ element: XCUIElement, toEqual expected: String, timeout: TimeInterval = UITestWait.standard) -> Bool {
    let predicate = NSPredicate(format: "label == %@", expected)
    let expectation = XCTNSPredicateExpectation(predicate: predicate, object: element)
    return XCTWaiter().wait(for: [expectation], timeout: timeout) == .completed
}

/// Taps `element`, then polls `condition` for the state change the tap is
/// expected to cause, re-tapping roughly once a second until `condition`
/// reports true or the overall bound elapses — `wait_until`, not a fixed
/// sleep, with a retry instead of a bare wait. A synthesized XCUITest touch
/// on a SwiftUI `Button` occasionally never reaches its action closure even
/// though the element reports `exists`/`isHittable` (observed here on a
/// glass control for its second tap in a row); retrying the tap is the
/// documented mitigation for that known XCUITest/SwiftUI gap. `condition`
/// must be a quick, non-blocking check (e.g. `element.exists`), not another
/// `waitForExistence` call.
@discardableResult
func tap(_ element: XCUIElement, until condition: @autoclosure () -> Bool, timeout: TimeInterval = UITestWait.standard) -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        element.tap()
        let stepDeadline = min(Date().addingTimeInterval(1), deadline)
        while Date() < stepDeadline {
            if condition() { return true }
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        }
    }
    return condition()
}
