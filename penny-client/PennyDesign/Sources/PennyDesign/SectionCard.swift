import SwiftUI

/// A section header ("Collections", "Turns", …). One shared view so every
/// page's headers sit on the same column with no page-specific offsets:
/// the header icon's left edge, `Tokens.sectionHeaderX`, derived from the
/// same tokens as the header grid.
public struct SectionHeader: View {
    public let title: String

    public init(title: String) {
        self.title = title
    }

    public var body: some View {
        Text(title)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .padding(.leading, Tokens.sectionHeaderInset)
    }
}

/// The ONLY way to make a section on the non-chat pages: the grey section
/// header, ONE white card, and the hairline dividers between the rows —
/// drawn here, never by the screens. Two shapes:
///
///  - a row card: `rows:` takes the rows (usually `ListRow`s, each wrapped
///    in its own NavigationLink where the row opens something) and inserts
///    a divider between every pair itself;
///  - a prose card: a `@ViewBuilder content` block (an entry's body, a
///    system prompt, a CodeBlock) padded inside the card.
public struct SectionCard<Content: View>: View {
    public let header: String?
    public var footer: String?
    var rows: [any View]
    var content: Content

    /// A prose card: the header and one white card around `content`.
    public init(header: String? = nil, footer: String? = nil,
                @ViewBuilder content: () -> Content) {
        self.header = header
        self.footer = footer
        self.rows = []
        self.content = content()
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let header {
                SectionHeader(title: header)
            }
            if rows.isEmpty {
                content
                    .padding(.horizontal, Tokens.cardInnerPadding)
                    .padding(.vertical, Tokens.margin)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(cardBackground)
            } else {
                VStack(spacing: 0) {
                    ForEach(rows.indices, id: \.self) { index in
                        if index > 0 { Divider() }
                        AnyView(rows[index])
                    }
                }
                .background(cardBackground)
            }
            if let footer {
                Text(footer)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .padding(.leading, Tokens.sectionHeaderInset)
            }
        }
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: Tokens.cardCorner, style: .continuous)
            .fill(Color(.secondarySystemGroupedBackground))
            .shadow(color: .black.opacity(0.06), radius: 8, y: 2)
    }
}

/// A row card: the header, one white card around `rows`, hairline dividers
/// between them, and an optional footer under the card.
extension SectionCard where Content == EmptyView {
    public init(header: String? = nil, footer: String? = nil, rows: [any View]) {
        self.header = header
        self.footer = footer
        self.rows = rows
        self.content = EmptyView()
    }
}
