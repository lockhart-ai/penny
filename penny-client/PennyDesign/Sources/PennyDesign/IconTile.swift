import SwiftUI

/// The Settings-app style leading icon: a white SF Symbol on a rounded
/// square of the token size (`Tokens.tile`), or of an explicit `side` —
/// symbol and corner radius scale with the side, so any size keeps the
/// same look.
public struct IconTile: View {
    public let symbol: String
    public let tint: Color
    public var side: CGFloat?

    public init(symbol: String, tint: Color, side: CGFloat? = nil) {
        self.symbol = symbol
        self.tint = tint
        self.side = side
    }

    public var body: some View {
        let tileSide = side ?? Tokens.tile
        RoundedRectangle(cornerRadius: tileSide * 0.22, style: .continuous)
            .fill(tint.gradient)
            .frame(width: tileSide, height: tileSide)
            .overlay {
                Image(systemName: symbol)
                    .font(.system(size: tileSide * 0.52, weight: .semibold))
                    .foregroundStyle(.white)
            }
    }
}
