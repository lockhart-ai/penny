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

/// Polls `element`'s accessibility value for an exact match (#3150 — a
/// chat message row's `.accessibilityValue` carries find-in-chat's
/// highlight state, `"highlighted-current"`/`"highlighted"`/empty) instead
/// of sleeping, the same reasoning as `waitForLabel` extended to `value`.
@discardableResult
func waitForValue(_ element: XCUIElement, toEqual expected: String, timeout: TimeInterval = UITestWait.standard) -> Bool {
    let predicate = NSPredicate(format: "value == %@", expected)
    let expectation = XCTNSPredicateExpectation(predicate: predicate, object: element)
    return XCTWaiter().wait(for: [expectation], timeout: timeout) == .completed
}

/// Polls for `element` to be gone — the non-existence counterpart to
/// `waitForExistence`, via the same `XCTNSPredicateExpectation` mechanism
/// `waitForLabel` uses, so a control that's already absent returns quickly
/// instead of blocking for the whole timeout (`waitForExistence(timeout:)`
/// itself can only return `false` by exhausting its timeout, since it keeps
/// polling for existence to become true until the deadline).
@discardableResult
func waitForAbsence(_ element: XCUIElement, timeout: TimeInterval = UITestWait.standard) -> Bool {
    let predicate = NSPredicate(format: "exists == 0")
    let expectation = XCTNSPredicateExpectation(predicate: predicate, object: element)
    return XCTWaiter().wait(for: [expectation], timeout: timeout) == .completed
}
