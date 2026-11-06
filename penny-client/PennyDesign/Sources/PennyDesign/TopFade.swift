import SwiftUI

/// The soft top fade under the nav row: content scrolling under the chrome
/// row dissolves into the page background. Opaque at its top edge, clear at
/// its bottom — the caller sizes it (the shell spans the whole chrome zone,
/// `Tokens.chromeTopGap + Tokens.navControlHeight + Tokens.topFadeBleed`,
/// so content placed at that clearance sits just below the fade).
public struct TopFade: View {
    /// The page background the content dissolves into — the app's
    /// `systemGroupedBackground`, handed in so the kit stays colour-neutral.
    let background: Color

    public init(background: Color) {
        self.background = background
    }

    public var body: some View {
        LinearGradient(
            colors: [background, background.opacity(0)],
            startPoint: .top,
            endPoint: .bottom
        )
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}
