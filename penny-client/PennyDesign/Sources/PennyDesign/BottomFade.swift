import SwiftUI

/// The soft bottom fade over the composer zone: content scrolling under
/// the message field dissolves into the page background, so nothing reads
/// behind the field's glass — the top fade's mirror (`TopFade`). Clear at
/// the band's top edge, fully opaque from `height - opaqueThrough` down —
/// i.e. opaque through the band's bottom `opaqueThrough` (the composer's
/// own extent, measured from the screen's bottom) — with the band's top
/// `height - opaqueThrough` as the dissolve ramp. The caller measures the
/// composer and sizes both from the tokens.
public struct BottomFade: View {
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
            .init(color: background.opacity(0), location: 0),
            .init(color: background, location: min((height - opaqueThrough) / max(height, 1), 1)),
            .init(color: background, location: 1)
        ], startPoint: .top, endPoint: .bottom)
            .frame(height: height)
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }
}
