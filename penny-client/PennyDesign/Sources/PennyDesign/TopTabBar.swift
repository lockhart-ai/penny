import SwiftUI

/// The four top-level tabs of the centred top tab bar.
public enum TopTab: CaseIterable, Identifiable {
    case chat
    case memory
    case prompts
    case settings

    public var id: Self { self }
}

/// The centred top tab bar: one glass capsule of four icon-only tabs, the
/// selected tab on a lighter inner pill, the Chat tab carrying the green
/// connected-dot badge, and an optional blue count badge on Prompts.
/// The Chat tab's avatar is the app's own — pass the avatar view (Penny's
/// circular image); the bar sizes it to 26pt and adds the connected dot.
public struct TopTabBar<Avatar: View>: View {
    public let selected: TopTab
    /// A live count badge shown on the Prompts tab (collectors running).
    public var promptsBadge: Int = 0
    /// What tapping a tab does.
    public var onSelect: (TopTab) -> Void
    let chatAvatar: Avatar

    public init(
        selected: TopTab,
        promptsBadge: Int = 0,
        onSelect: @escaping (TopTab) -> Void = { _ in },
        @ViewBuilder chatAvatar: () -> Avatar
    ) {
        self.selected = selected
        self.promptsBadge = promptsBadge
        self.onSelect = onSelect
        self.chatAvatar = chatAvatar()
    }

    public var body: some View {
        HStack(spacing: 0) {
            ForEach(TopTab.allCases) { tab in
                Button {
                    onSelect(tab)
                } label: {
                    icon(for: tab)
                        // Content row of the 44pt capsule: minus the glass's 4pt inset each side.
                        .frame(width: 44, height: Tokens.navControlHeight - 8)
                        .background {
                            if selected == tab {
                                Capsule()
                                    .fill(Color.white.opacity(0.9))
                                    .frame(width: 40, height: 30)
                            }
                        }
                        .overlay(alignment: .topTrailing) {
                            if tab == .prompts, promptsBadge > 0 {
                                Text("\(promptsBadge)")
                                    .font(.caption2.weight(.bold).monospacedDigit())
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 1)
                                    .background(Capsule().fill(.blue))
                                    .offset(x: 7, y: -5)
                            }
                        }
                        .foregroundStyle(selected == tab ? .primary : .secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(4)
        .glassEffect(.regular, in: .capsule)
    }

    @ViewBuilder
    private func icon(for tab: TopTab) -> some View {
        switch tab {
        case .chat:
            chatAvatar
                .frame(width: 26, height: 26)
                .overlay(alignment: .bottomTrailing) {
                    ZStack {
                        Circle().fill(.white)
                        Circle().fill(.green).frame(width: 8, height: 8)
                    }
                    .frame(width: 11, height: 11)
                    .offset(x: 2, y: 2)
                }
        case .memory:
            Image(systemName: "books.vertical")
                .font(.system(size: 17, weight: .semibold))
        case .prompts:
            Image(systemName: "list.bullet.rectangle")
                .font(.system(size: 17, weight: .semibold))
        case .settings:
            Image(systemName: "gearshape")
                .font(.system(size: 17, weight: .semibold))
        }
    }
}

/// The app's ONE glass circle: `Tokens.navControlHeight` (44pt) side, the nav
/// circles' glyph size, glass in a circle. Every glass circle in the app is
/// this shape — the nav row's search and back circles, the page headers'
/// trailing buttons, find-in-chat's step and close circles — so they all
/// read as one size on one grid.
public struct GlassCircle: View {
    public let symbol: String
    /// Dimmed and inert: find-in-chat's step circles before any match.
    public var dimmed: Bool = false

    public init(symbol: String, dimmed: Bool = false) {
        self.symbol = symbol
        self.dimmed = dimmed
    }

    public var body: some View {
        Image(systemName: symbol)
            .font(.system(size: Tokens.navGlyphSize, weight: .semibold))
            .foregroundStyle(dimmed ? AnyShapeStyle(.tertiary) : AnyShapeStyle(.primary))
            .frame(width: Tokens.navControlHeight, height: Tokens.navControlHeight)
            .glassEffect(.regular, in: .circle)
    }
}

/// The tappable glass circle button. The label-only `GlassCircle` is what a
/// header button riding a `NavigationLink` uses, so the link keeps the tap.
public struct GlassCircleButton: View {
    public let symbol: String
    public var dimmed: Bool = false
    public var onTap: () -> Void = {}

    public init(symbol: String, dimmed: Bool = false, onTap: @escaping () -> Void = {}) {
        self.symbol = symbol
        self.dimmed = dimmed
        self.onTap = onTap
    }

    public var body: some View {
        Button {
            onTap()
        } label: {
            GlassCircle(symbol: symbol, dimmed: dimmed)
        }
        .buttonStyle(.plain)
        .disabled(dimmed)
    }
}
