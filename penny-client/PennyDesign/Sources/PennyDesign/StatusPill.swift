import SwiftUI

/// A small capsule: caption2 bold uppercase, coloured text on the same
/// colour at 0.15 opacity. Fixed width where text follows it.
public struct StatusPill: View {
    public let label: String
    public let color: Color
    public var symbol: String?
    public var width: CGFloat?

    public init(status: RunStatus) {
        self.init(label: status.rawValue, color: status.color, symbol: status.symbol)
    }

    public init(label: String, color: Color, symbol: String? = nil, width: CGFloat? = nil) {
        self.label = label
        self.color = color
        self.symbol = symbol
        self.width = width
    }

    public var body: some View {
        HStack(spacing: 3) {
            if let symbol {
                Image(systemName: symbol)
                    .font(.system(size: 8, weight: .bold))
            }
            Text(label)
                .font(.caption2.weight(.bold))
                .kerning(0.3)
        }
        .foregroundStyle(color)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(color.opacity(0.15), in: Capsule())
        .frame(width: width, alignment: .center)
    }
}
