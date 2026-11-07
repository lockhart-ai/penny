import SwiftUI

/// The app's one glass circle: `Tokens.navControlHeight` (44pt) side, the
/// nav glyph size, glass in a circle. Its own real callers — the shell's
/// hand-built back/search circles — retired with the stock `NavigationStack`
/// shell (#3111); it stays because `KitPageHeaderStory` still demos a
/// `GlassCircleButton` in `PageTitle`'s trailing slot.
public struct GlassCircle: View {
    public let symbol: String
    /// Dimmed and inert.
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

/// The tappable glass circle button.
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
