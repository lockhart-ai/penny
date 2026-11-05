import SwiftUI

/// The five metric columns at fixed token widths, so values line up card to
/// card and row to row. caption2 (caption wrapped the cached column).
public struct MetricsStrip: View {
    public let values: [String]

    private static let symbols = ["arrow.down", "square.stack.3d.down.right", "arrow.up", "speedometer", "timer"]

    public init(values: [String]) {
        self.values = values
    }

    public var body: some View {
        HStack(spacing: Tokens.metricSpacing) {
            ForEach(Array(zip(zip(Self.symbols, Tokens.metricColumns), values)), id: \.0.0) { pair, value in
                HStack(spacing: 3) {
                    Image(systemName: pair.0)
                        .font(.system(size: 9, weight: .semibold))
                    Text(value)
                        .lineLimit(1)
                        .fixedSize()
                }
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: pair.1, alignment: .leading)
            }
        }
    }
}
