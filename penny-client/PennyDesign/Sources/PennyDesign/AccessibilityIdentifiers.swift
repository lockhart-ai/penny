/// The app's one set of accessibility identifiers (#3106): every control
/// `PennyClientUITests` drives, or screen it asserts on, is named here once
/// and read from here on both sides — the view that sets it and the test
/// that looks it up — so no test and no view ever carries its own copy of
/// the same string.
public enum AXID {
    /// A tab's button in the top capsule (`TopTabBar`).
    public static func tabButton(_ tab: TopTab) -> String { "shell.tab.button.\(tab.rawValue)" }
    /// The root of a tab's content under the shell (`AppShellView.tabContent`).
    public static func tabRoot(_ tab: TopTab) -> String { "shell.tab.root.\(tab.rawValue)" }
    /// The glass back circle shown over a pushed page.
    public static let backButton = "shell.back"
    /// The glass circle that opens find-in-chat.
    public static let searchOpenButton = "shell.search.open"

    /// Find-in-chat's query field.
    public static let searchField = "chat.search.field"
    /// Find-in-chat's "n of m" match counter.
    public static let searchCounter = "chat.search.counter"
    /// Find-in-chat's step-to-older-match circle (chevron up).
    public static let searchOlderButton = "chat.search.older"
    /// Find-in-chat's step-to-newer-match circle (chevron down).
    public static let searchNewerButton = "chat.search.newer"
    /// Find-in-chat's close circle.
    public static let searchCloseButton = "chat.search.close"

    /// The composer's text field.
    public static let composerTextField = "composer.textField"
    /// The composer's send arrow — only present while there's a draft.
    public static let composerSendButton = "composer.send"

    /// A memory's row in the Memory tab's list, keyed by its name.
    public static func memoryRow(_ name: String) -> String { "memory.row.\(name)" }
    /// The root of a memory's pushed detail page.
    public static let memoryDetailRoot = "memory.detail.root"
}
