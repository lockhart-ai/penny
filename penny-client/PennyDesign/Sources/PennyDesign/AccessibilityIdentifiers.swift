/// The four pages the shell's leading page `Menu` switches between
/// (#3111) — one stock `NavigationStack`, root content swapped by this
/// selection; no more per-page tab buttons. `String`-backed so each case
/// has a stable raw value, the one source `AXID.tabRoot` builds its
/// accessibility identifier from.
public enum TopTab: String, CaseIterable, Identifiable {
    case chat
    case memory
    case prompts
    case settings

    public var id: Self { self }

    /// The page menu row's label and the closed menu icon alike — mirrors
    /// the approved mock's `ChatPageSection`.
    public var name: String {
        switch self {
        case .chat: return "Chat"
        case .memory: return "Memory"
        case .prompts: return "Prompts"
        case .settings: return "Settings"
        }
    }

    /// The outline symbol every page uses, closed and open alike (the
    /// code owner approved the outline bubble, not `.fill`, for Chat).
    public var symbol: String {
        switch self {
        case .chat: return "bubble.left"
        case .memory: return "books.vertical"
        case .prompts: return "list.bullet.rectangle"
        case .settings: return "gearshape"
        }
    }
}

/// The app's one set of accessibility identifiers (#3106): every control
/// `PennyClientUITests` drives, or screen it asserts on, is named here once
/// and read from here on both sides — the view that sets it and the test
/// that looks it up — so no test and no view ever carries its own copy of
/// the same string.
public enum AXID {
    /// The root of a page's content under the shell (`AppShellView.content(for:)`).
    public static func tabRoot(_ tab: TopTab) -> String { "shell.tab.root.\(tab.rawValue)" }
    /// The leading page menu (#3111) — closed, its label is the current
    /// page's icon; open, a `Picker` row per page.
    public static let pageMenu = "shell.page.menu"

    /// The composer's text field.
    public static let composerTextField = "composer.textField"
    /// The composer's send arrow — only present while there's a draft.
    public static let composerSendButton = "composer.send"

    /// A memory's row in the Memory page's list, keyed by its name.
    public static func memoryRow(_ name: String) -> String { "memory.row.\(name)" }
    /// The root of a memory's pushed detail page.
    public static let memoryDetailRoot = "memory.detail.root"

    /// A chat message row, keyed by its id (#3150) — its `.accessibilityValue`
    /// carries find-in-chat's highlight state (`"highlighted-current"`,
    /// `"highlighted"`, or empty), a non-pixel signal that typing in search
    /// actually marked the matching bubble, not just that the message's own
    /// text (always present, search or not) happened to be on screen.
    public static func chatMessageRow(_ id: Int) -> String { "chat.message.row.\(id)" }
}
