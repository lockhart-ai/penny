import SwiftUI

/// A kind of thing a row is about: the shared type kinds carry their own
/// symbol and colour — icon tiles are never set per item.
public protocol RowIconKind {
    var symbol: String { get }
    var color: Color { get }
}

extension RunKind: RowIconKind {}
extension MemoryKind: RowIconKind {}
extension TurnRole: RowIconKind {}

/// A list row's leading tile: a type kind's tile, or an explicit tile with
/// an optional badge (a turn number riding the tile's corner).
public struct ListRowTile {
    public let symbol: String
    public let tint: Color
    public var badge: Int?

    /// The tile of a type kind — symbol and tint come from the kind.
    public static func kind(_ kind: some RowIconKind) -> Self {
        Self(symbol: kind.symbol, tint: kind.color)
    }

    /// An explicit tile.
    public static func tile(symbol: String, tint: Color, badge: Int? = nil) -> Self {
        Self(symbol: symbol, tint: tint, badge: badge)
    }
}

/// The app's ONE list row. Screens pass data — icon, title, pill, trailing
/// item, subtitle, caption, chevron, progress — and the widget picks every
/// font, spacing and alignment itself, so no screen builds its own row and
/// alignment cannot drift between pages. The rule, literal: the tile's TOP
/// edge is level with the TOP OF THE CAPITAL LETTERS of the title (the cap
/// height of the font the widget chose), computed from the real font
/// metrics like `PageTitle`'s header tile. The pill, the trailing item and
/// the chevron all sit on the title line; the subtitle and caption hang
/// under it in the same text column. Paddings, the tile's x and the text
/// column all come from the tokens (`Tokens.cardInnerPadding`, `Tokens.tile`,
/// `Tokens.iconTextGap`, `Tokens.rowTextX`, `Tokens.rowVerticalPadding`).
public struct ListRow: View {
    /// What rides the title line's trailing edge.
    public enum Trailing {
        case none
        /// A timestamp: caption, monospaced digits, secondary.
        case time(String)
        /// A value: subheadline, secondary.
        case value(String)
        case toggle(Binding<Bool>)
        /// The working ellipsis — a turn still running.
        case running
    }

    /// The row's second line, standard or monospaced.
    public enum Subtitle {
        case standard(String)
        case monospaced(String)
    }

    /// The row's caption line: a text with an optional trailing text, or
    /// the metrics strip.
    public enum Caption {
        case text(String, trailing: String?)
        case metrics([String])

        public static func text(_ text: String) -> Self { .text(text, trailing: nil) }
    }

    public var icon: ListRowTile?
    public let title: String
    public var pill: StatusPill?
    public var trailing: Trailing = .none
    public var subtitle: Subtitle?
    public var caption: Caption?
    public var showsChevron: Bool = false
    /// A live row's progress fraction along its bottom edge, filling from
    /// the leading edge.
    public var progress: Double?

    public init(
        icon: ListRowTile? = nil,
        title: String,
        pill: StatusPill? = nil,
        trailing: Trailing = .none,
        subtitle: Subtitle? = nil,
        caption: Caption? = nil,
        showsChevron: Bool = false,
        progress: Double? = nil
    ) {
        self.icon = icon
        self.title = title
        self.pill = pill
        self.trailing = trailing
        self.subtitle = subtitle
        self.caption = caption
        self.showsChevron = showsChevron
        self.progress = progress
    }

    /// The title's style: headline when anything rides under the title,
    /// body for title-only rows.
    private var titleStyle: Font.TextStyle {
        (subtitle != nil || caption != nil) ? .headline : .body
    }
    /// The weight each title style renders at — the font the cap-height
    /// alignment measures and the font the text uses.
    private var titleWeight: Font.Weight {
        titleStyle == .headline ? .semibold : .regular
    }
    /// The title's cap height: the distance from the title line's baseline
    /// up to the top of its capital letters, in the chosen font.
    private var cap: CGFloat { capHeight(titleStyle, titleWeight) }

    public var body: some View {
        let titleCap = cap
        return HStack(alignment: .firstTextBaseline, spacing: Tokens.iconTextGap) {
            if let icon {
                tile(for: icon)
                    .alignmentGuide(.firstTextBaseline) { _ in titleCap }
            }
            VStack(alignment: .leading, spacing: Tokens.rowLineGap) {
                titleLine
                if let subtitle {
                    switch subtitle {
                    case .standard(let text):
                        Text(text)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    case .monospaced(let text):
                        Text(text)
                            .font(.subheadline.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
                if let caption {
                    switch caption {
                    case .text(let text, let trailing):
                        HStack(alignment: .firstTextBaseline, spacing: 6) {
                            Text(text)
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                                .lineLimit(1)
                            Spacer(minLength: 8)
                            if let trailing {
                                Text(trailing)
                                    .font(.caption)
                                    .foregroundStyle(.tertiary)
                                    .lineLimit(1)
                            }
                        }
                    case .metrics(let values):
                        MetricsStrip(values: values)
                    }
                }
            }
        }
        .padding(.horizontal, Tokens.cardInnerPadding)
        .padding(.vertical, Tokens.rowVerticalPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
        .overlay(alignment: .bottom) { progressEdge }
    }

    /// The title line: the title, the pill and the running ellipsis after
    /// it, then the trailing item and the chevron — all on one baseline.
    private var titleLine: some View {
        let titleCap = cap
        return HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(title)
                .font(Font.system(titleStyle).weight(titleWeight))
                .lineLimit(2)
            if let pill {
                pill
            }
            if case .running = trailing {
                Image(systemName: "ellipsis")
                    .symbolEffect(.variableColor.iterative)
                    .foregroundStyle(.blue)
                    .font(.footnote)
            }
            Spacer(minLength: 8)
            switch trailing {
            case .none:
                EmptyView()
            case .time(let time):
                Text(time)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            case .value(let value):
                Text(value)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            case .toggle(let isOn):
                // The toggle centers on the title's cap-height band: the
                // point cap/2 above its bottom sits on the title's
                // baseline, so its middle rides midway up the capital band.
                Toggle("", isOn: isOn)
                    .labelsHidden()
                    .alignmentGuide(.firstTextBaseline) { dimensions in
                        dimensions.height / 2 + titleCap / 2
                    }
            case .running:
                EmptyView()
            }
            if showsChevron {
                Image(systemName: "chevron.right")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    /// The leading tile, with its optional badge riding the corner.
    @ViewBuilder private func tile(for icon: ListRowTile) -> some View {
        IconTile(symbol: icon.symbol, tint: icon.tint)
            .overlay(alignment: .topTrailing) {
                if let badge = icon.badge {
                    Text("\(badge)")
                        .font(.system(size: 9, weight: .bold).monospacedDigit())
                        .foregroundStyle(.white)
                        .frame(width: 15, height: 15)
                        .background(Circle().fill(Color(.systemGray)))
                        .background(Circle().fill(.white))
                        .offset(x: 5, y: -5)
                }
            }
    }

    /// A live row's thin blue progress bar along the bottom edge, filling
    /// from the leading edge.
    @ViewBuilder private var progressEdge: some View {
        if let progress {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.blue.opacity(0.25))
                    Capsule().fill(Color.blue)
                        .frame(width: geo.size.width * progress)
                }
            }
            .frame(height: 2)
            .padding(.horizontal, 10)
            .padding(.bottom, 2)
        }
    }
}
