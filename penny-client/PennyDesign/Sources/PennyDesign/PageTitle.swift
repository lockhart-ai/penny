import SwiftUI

/// A page's title block, the same on every non-chat page: the page's type
/// icon tile, the required byline under the title, and optional trailing
/// glass buttons. Everything sits on the one derived grid in Tokens: the
/// title and byline start exactly at `Tokens.rowTextX` — the rows' text
/// column — and the tile ends `Tokens.iconTextGap` before that, so its left
/// edge is at `Tokens.pageIconX`, the column the section headers start on
/// too. The tile's side is the fixed `Tokens.pageIconSize` — the span from
/// the title's cap top (its top edge, by the baseline guide) down to the
/// byline's baseline — so the header is identical on every page; the title
/// is one line (`.lineLimit(1)`), never measured from a wrapped layout. The
/// caller supplies the screen margin's horizontal padding; the block adds
/// only what remains to reach the tile's column.
public struct PageTitle<Trailing: View>: View {
    public let title: String
    public let byline: String
    public let symbol: String
    public let tint: Color
    /// An optional status pill riding the byline line (a live run's
    /// RUNNING), after the byline text.
    public var bylinePill: StatusPill?
    var trailing: Trailing

    public init(
        title: String,
        byline: String,
        symbol: String,
        tint: Color,
        bylinePill: StatusPill? = nil,
        @ViewBuilder trailing: () -> Trailing = { EmptyView() }
    ) {
        self.title = title
        self.byline = byline
        self.symbol = symbol
        self.tint = tint
        self.bylinePill = bylinePill
        self.trailing = trailing()
    }

    public var body: some View {
        let cap = capHeight(Tokens.pageTitleStyle, Tokens.pageTitleWeight)
        HStack(alignment: .firstTextBaseline, spacing: Tokens.iconTextGap) {
            // The tile spans the header: its guide point sits `cap` below
            // its top (the cap top), so the top edge lands on the cap top
            // and the bottom edge — one fixed `Tokens.pageIconSize` lower —
            // lands on the byline's baseline.
            IconTile(symbol: symbol, tint: tint, side: Tokens.pageIconSize)
                .alignmentGuide(.firstTextBaseline) { _ in cap }
            VStack(alignment: .leading, spacing: Tokens.pageTitleGap) {
                Text(title)
                    .font(Font.system(Tokens.pageTitleStyle).weight(Tokens.pageTitleWeight))
                    .lineLimit(1)
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(byline)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    if let bylinePill {
                        bylinePill
                    }
                }
            }
            Spacer(minLength: 8)
            // The glass button top-aligns with the cap top, not the line box.
            trailing
                .alignmentGuide(.firstTextBaseline) { dimensions in dimensions[.top] + cap }
        }
        // The tile's left edge lands on `Tokens.pageIconX`: the caller's
        // horizontal padding supplies the screen margin, this the rest.
        .padding(.leading, Tokens.pageTitleLeadingInset)
    }
}
