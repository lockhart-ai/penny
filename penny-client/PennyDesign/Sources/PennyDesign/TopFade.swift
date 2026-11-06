import SwiftUI

/// The soft top fade under the nav row: content scrolling under the chrome
/// row dissolves into the page background, so nothing reads behind the
/// row's glass. Fully opaque from its top edge through `opaqueThrough` (the
/// chrome row's bottom, from the screen's top) and clear at `height` —
/// the mock's `ChatCanvasTopFade` shape. The caller measures the top
/// safe-area inset and sizes both from the tokens.
public struct TopFade: View {
    /// The page background the content dissolves into — the app's
    /// `systemGroupedBackground`, handed in so the kit stays colour-neutral.
    let background: Color
    let height: CGFloat
    let opaqueThrough: CGFloat

    public init(background: Color, height: CGFloat, opaqueThrough: CGFloat) {
        self.background = background
        self.height = height
        self.opaqueThrough = opaqueThrough
    }

    public var body: some View {
        LinearGradient(stops: [
            .init(color: background, location: 0),
            .init(color: background, location: min(opaqueThrough / max(height, 1), 1)),
            .init(color: background.opacity(0), location: 1)
        ], startPoint: .top, endPoint: .bottom)
            .frame(height: height)
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }
}
