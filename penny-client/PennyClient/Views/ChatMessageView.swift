import Foundation
import SwiftUI
import PennyDesign
import PennyServices

/// The chat tab's message row. The `.message` layout is the redesigned
/// bubble (`MessageBubble` in PennyDesign): Penny in solid white — her
/// image riding the top edge to edge — and the user in solid blue, each
/// with the metadata line under it. Markdown, tables and links are parsed
/// here and handed to the kit bubble as rendered content; a table wider
/// than the bubble scrolls horizontally inside it. The `.compact` and
/// `.media` layouts rendered the old semantic search screen's result
/// cards; that screen is gone (#3045 — find-in-chat replaced it) and
/// nothing constructs these two layouts anymore — left in place rather
/// than cleaned up in the same change, flagged as a follow-up.
struct ChatMessageView: View {
    let message: ChatMessage
    let layout: MessageView.MessageLayout
    /// Whether this is a message sent while Penny is working (#3043) —
    /// renders at `Tokens.queuedBubbleOpacity` with "🕒 Queued" standing
    /// in for the real time, until the view model clears the mark.
    let isQueued: Bool
    /// Find-in-chat (#3045): the query to highlight in this row's text, and
    /// whether this message is the current match — every occurrence of the
    /// query gets a solid background (dark text on both bubble colours),
    /// the current message's occurrences a stronger orange. Matches are
    /// counted per message, not per occurrence, so every occurrence in the
    /// current message's bubble is marked, not just one. The search itself
    /// matches a message's full stored content, but highlighting only
    /// reaches what's actually rendered as this row's text blocks — a
    /// table's cells and an outgoing reply's quoted block (rendered
    /// separately, by the kit's `MessageBubble`) are known gaps: such a
    /// message can count as a match without a visible highlight on it.
    /// `nil`/`false` outside a search.
    var searchQuery: String?
    var isCurrentSearchMatch = false

    init(
        message: ChatMessage,
        layout: MessageView.MessageLayout,
        isQueued: Bool = false,
        searchQuery: String? = nil,
        isCurrentSearchMatch: Bool = false
    ) {
        self.message = message
        self.layout = layout
        self.isQueued = isQueued
        self.searchQuery = searchQuery
        self.isCurrentSearchMatch = isCurrentSearchMatch
    }

    private var messageContentBlocks: [ChatMessageContentBlock] {
        ChatMessageContentParser.parse(message.content)
    }

    private var replyContent: ChatReplyContent? {
        ChatReplyContentParser.parse(message.content)
    }

    private var sourceTitle: String {
        if let sourceHint = message.sourceHint, !sourceHint.isEmpty {
            return sourceHint
        }
        return message.isOutgoing ? "You" : "Penny"
    }

    private var firstAttachment: ImageAttachment? {
        message.imageAttachments.first
    }

    private var compactCardBackground: Color {
        message.isOutgoing ? Color.accentColor : Color(.secondarySystemGroupedBackground)
    }

    private var compactPrimaryForeground: Color {
        message.isOutgoing ? .white : .primary
    }

    private var compactSecondaryForeground: Color {
        message.isOutgoing ? .white.opacity(0.78) : .secondary
    }

    private func attachmentThumbnail(_ attachment: ImageAttachment, height: CGFloat, cornerRadius: CGFloat) -> some View {
        GeometryReader { proxy in
            Image(uiImage: attachment.image)
                .resizable()
                .scaledToFill()
                .frame(width: proxy.size.width, height: height)
                .clipped()
        }
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }

    /// A table wider than the bubble scrolls horizontally inside it —
    /// never overflowing the bubble or squashing below its minimum
    /// column widths. Messages with a table stretch their bubble to the
    /// row so the scroll view has a width to scroll against.
    private var hasTableBlock: Bool {
        messageContentBlocks.contains { block in
            if case .table = block.kind { return true }
            return false
        }
    }

    /// The run's duration in seconds, as the metadata line shows it.
    static func durationLabel(_ durationMs: Int?) -> String? {
        guard let durationMs else { return nil }
        let seconds = Int((Double(durationMs) / 1000).rounded())
        return "\(seconds) s"
    }

    /// The message row: the redesigned kit bubble with the metadata line
    /// under it. The quote is one-way — only the user's outgoing reply
    /// carries one (`ChatReplyContentParser` reads it back out of the
    /// message's own text); Penny's bubbles never show a quote. A
    /// collector or notification message names its source as the metadata
    /// line's first segment; chat replies carry no source. `isQueued`
    /// (#3043) overrides the time with "🕒 Queued" and fades the whole
    /// bubble — the message itself is unchanged, already sent.
    private var messageRow: some View {
        MessageBubble(
            role: message.isOutgoing ? .user : .penny,
            images: message.imageAttachments.map(\.image),
            quote: message.isOutgoing ? replyContent?.summary : nil,
            quoteAuthor: replyContent?.author ?? "Penny",
            fillsWidth: hasTableBlock,
            time: isQueued ? "🕒 Queued" : message.displayTime,
            source: message.isOutgoing ? nil : message.sourceHint,
            tools: message.toolCalls,
            duration: Self.durationLabel(message.durationMs)
        ) {
            if message.isOutgoing {
                outgoingMessageText
            } else if !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                messageContent
            }
        }
        // `.contain` first: without it, `.accessibilityIdentifier` broadcasts
        // onto every leaf element inside the bubble (its text, time label,
        // tool summary, …) instead of naming the row as one element — the
        // same reasoning `AppShellView.content(for:)` documents for a page
        // root.
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier(AXID.chatMessageRow(message.id))
        .accessibilityValue(searchHighlightAccessibilityValue)
    }

    /// A non-pixel signal that find-in-chat's highlight actually reached
    /// this bubble (#3150) — the message's own text being on screen proves
    /// nothing by itself, since it's there with or without a search (a UI
    /// test asserting only that proved weak: it would pass even if
    /// highlighting silently broke). Empty outside a match, so the common
    /// case carries no extra accessibility noise.
    private var searchHighlightAccessibilityValue: String {
        guard let searchQuery, !searchQuery.isEmpty,
              message.content.localizedCaseInsensitiveContains(searchQuery) else { return "" }
        return isCurrentSearchMatch ? "highlighted-current" : "highlighted"
    }

    @ViewBuilder
    private var outgoingMessageText: some View {
        if let replyContent {
            highlightedText(replyContent.body)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            highlightedText(message.content)
        }
    }

    /// Plain text (the user's bubble carries no markdown), with find-in-chat
    /// highlighting applied when a search is active.
    private func highlightedText(_ text: String) -> Text {
        guard let searchQuery else {
            return Text(text).font(.body)
        }
        return Text(ChatMessageContentParser.highlighted(
            AttributedString(text), query: searchQuery, isCurrentMatch: isCurrentSearchMatch
        )).font(.body)
    }

    private var messageContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(messageContentBlocks) { block in
                messageContentBlock(block)
            }
        }
        .lineLimit(nil)
        .font(.body)
        .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    private func messageContentBlock(_ block: ChatMessageContentBlock) -> some View {
        switch block.kind {
        case .text(let text):
            ChatAttributedText(attributedText: highlightedBlockText(text))
                .lineLimit(nil)
                .font(.body)
        case .table(let table):
            // Find-in-chat highlighting doesn't reach table cells (#3045).
            ScrollView(.horizontal, showsIndicators: false) {
                ChatMarkdownTableView(table: table)
            }
        }
    }

    private func highlightedBlockText(_ text: AttributedString) -> AttributedString {
        guard let searchQuery else { return text }
        return ChatMessageContentParser.highlighted(text, query: searchQuery, isCurrentMatch: isCurrentSearchMatch)
    }

    var body: some View {
        switch layout {
        case .message:
            messageRow
                .opacity(isQueued ? Tokens.queuedBubbleOpacity : 1)
        case .compact:
            compactCard
        case .media:
            mediaCard
        }
    }

    private var compactCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let firstAttachment {
                attachmentThumbnail(firstAttachment, height: 96, cornerRadius: 8)
            } else if message.isOutgoing {
                Color.clear
                    .frame(maxWidth: .infinity)
                    .frame(height: 96)
            }

            Text(sourceTitle)
                .font(.caption.weight(.semibold))
                .foregroundStyle(compactSecondaryForeground)
                .lineLimit(1)
                .frame(height: 14, alignment: .top)

            Text(message.content)
                .font(.subheadline)
                .foregroundStyle(compactPrimaryForeground)
                .lineLimit(3)
                .frame(maxWidth: .infinity, minHeight: 54, alignment: .topLeading)
        }
        .padding(8)
        .frame(maxWidth: .infinity, minHeight: 192, alignment: .topLeading)
        .background(compactCardBackground, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var mediaCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            mediaBubble

            Text(sourceTitle)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private var mediaBubble: some View {
        Group {
            if let firstAttachment {
                attachmentThumbnail(firstAttachment, height: 86, cornerRadius: 8)
            } else {
                Image(systemName: "photo")
                    .font(.title2)
                    .foregroundStyle(message.isOutgoing ? .white.opacity(0.78) : Color(.tertiaryLabel))
                    .frame(maxWidth: .infinity)
                    .frame(height: 86)
                    .background(message.isOutgoing ? Color.accentColor : Color(.tertiarySystemGroupedBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
        }
        .padding(message.isOutgoing ? 3 : 0)
        .background(message.isOutgoing ? Color.accentColor : Color.clear, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

struct ChatAttributedText: View {
    let attributedText: AttributedString

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                ChatAttributedTextLine(attributedText: line)
            }
        }
    }

    private var lines: [AttributedString] {
        // A newline inside one Text view cannot move its sibling link/title views.
        // Give each explicit line its own flow and keep empty lines at text height.
        attributedText.characters.split(separator: "\n", omittingEmptySubsequences: false).map { line in
            line.isEmpty ? AttributedString(" ") : AttributedString(attributedText[line.startIndex..<line.endIndex])
        }
    }
}

private struct ChatAttributedTextLine: View {
    let attributedText: AttributedString

    var body: some View {
        ChatInlineFlowLayout(horizontalSpacing: 0, verticalSpacing: 0) {
            ForEach(fragments) { fragment in
                switch fragment.kind {
                case .text(let text):
                    Text(text)
                        .fixedSize(horizontal: false, vertical: true)
                case .compactLink(let title, let url):
                    Link(destination: url) {
                        HStack(alignment: .firstTextBaseline, spacing: 3) {
                            Text(title)
                            Image(systemName: "link")
                                .imageScale(.small)
                                .accessibilityHidden(true)
                        }
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.tint)
                }
            }
        }
    }

    private var fragments: [Fragment] {
        ChatMessageContentParser.inlineSegments(in: attributedText).enumerated().map(fragment(at:segment:))
    }

    private func fragment(at id: Int, segment: ChatMessageContentParser.InlineSegment) -> Fragment {
        switch segment {
        case .text(let text):
            return Fragment(id: id, kind: .text(text))
        case .link(let title, let url):
            return Fragment(id: id, kind: .compactLink(title: title, url: url))
        }
    }

    private struct Fragment: Identifiable {
        let id: Int
        let kind: Kind

        enum Kind {
            case text(AttributedString)
            case compactLink(title: String, url: URL)
        }
    }
}

private struct ChatInlineFlowLayout: Layout {
    let horizontalSpacing: CGFloat
    let verticalSpacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let availableWidth = proposal.width ?? .greatestFiniteMagnitude
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0
        var measuredWidth: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(ProposedViewSize(width: availableWidth, height: proposal.height))

            if currentX > 0, currentX + size.width > availableWidth {
                measuredWidth = max(measuredWidth, currentX - horizontalSpacing)
                currentX = 0
                currentY += lineHeight + verticalSpacing
                lineHeight = 0
            }

            currentX += min(size.width, availableWidth) + horizontalSpacing
            lineHeight = max(lineHeight, size.height)
        }

        measuredWidth = max(measuredWidth, currentX > 0 ? currentX - horizontalSpacing : 0)
        return CGSize(width: min(measuredWidth, availableWidth), height: currentY + lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let availableWidth = bounds.width
        var currentX = bounds.minX
        var currentY = bounds.minY
        var lineHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(ProposedViewSize(width: availableWidth, height: proposal.height))

            if currentX > bounds.minX, currentX + size.width > bounds.maxX {
                currentX = bounds.minX
                currentY += lineHeight + verticalSpacing
                lineHeight = 0
            }

            subview.place(
                at: CGPoint(x: currentX, y: currentY),
                anchor: .topLeading,
                proposal: ProposedViewSize(width: min(size.width, availableWidth), height: size.height)
            )
            currentX += min(size.width, availableWidth) + horizontalSpacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}

struct ChatMessageContentBlock: Identifiable {
    let id: Int
    let kind: Kind

    enum Kind {
        case text(AttributedString)
        case table(ChatMarkdownTable)
    }
}

struct ChatMarkdownTable {
    let headers: [String]
    let alignments: [ColumnAlignment]
    let rows: [[String]]

    var columnCount: Int {
        max(headers.count, rows.map(\.count).max() ?? 0)
    }

    func alignment(for columnIndex: Int) -> ColumnAlignment {
        alignments.indices.contains(columnIndex) ? alignments[columnIndex] : .leading
    }

    enum ColumnAlignment {
        case leading
        case center
        case trailing

        var frameAlignment: Alignment {
            switch self {
            case .leading:
                return .topLeading
            case .center:
                return .top
            case .trailing:
                return .topTrailing
            }
        }

        var textAlignment: TextAlignment {
            switch self {
            case .leading:
                return .leading
            case .center:
                return .center
            case .trailing:
                return .trailing
            }
        }
    }
}

private struct ChatMarkdownTableView: View {
    let table: ChatMarkdownTable

    var body: some View {
        ChatMarkdownTableLayout(columnCount: table.columnCount) {
            ForEach(0..<table.columnCount, id: \.self) { columnIndex in
                tableCell(
                    table.headers.indices.contains(columnIndex) ? table.headers[columnIndex] : "",
                    columnIndex: columnIndex,
                    rowIndex: 0,
                    isHeader: true
                )
            }

            ForEach(table.rows.indices, id: \.self) { rowIndex in
                ForEach(0..<table.columnCount, id: \.self) { columnIndex in
                    let cell = table.rows[rowIndex].indices.contains(columnIndex) ? table.rows[rowIndex][columnIndex] : ""
                    tableCell(cell, columnIndex: columnIndex, rowIndex: rowIndex + 1, isHeader: false)
                }
            }
        }
        .background(Color(.separator).opacity(0.35))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private func tableCell(_ value: String, columnIndex: Int, rowIndex: Int, isHeader: Bool) -> some View {
        let attributedValue = ChatMessageContentParser.markdownText(from: value.isEmpty ? " " : value)
        let alignment = table.alignment(for: columnIndex)

        return ChatAttributedText(attributedText: attributedValue)
            .font(isHeader ? .subheadline.weight(.semibold) : .subheadline)
            .multilineTextAlignment(alignment.textAlignment)
            .lineLimit(nil)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: alignment.frameAlignment)
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(isHeader ? Color(.tertiarySystemGroupedBackground) : Color(.secondarySystemGroupedBackground))
            .overlay(alignment: .bottom) {
                Rectangle()
                    .fill(Color(.separator).opacity(0.35))
                    .frame(height: 0.5)
            }
            .overlay(alignment: .trailing) {
                Rectangle()
                    .fill(Color(.separator).opacity(0.35))
                    .frame(width: 0.5)
            }
            .layoutValue(key: ChatMarkdownTableRowKey.self, value: rowIndex)
            .layoutValue(key: ChatMarkdownTableColumnKey.self, value: columnIndex)
    }
}

private struct ChatMarkdownTableLayout: Layout {
    let columnCount: Int

    private let minimumColumnWidth: CGFloat = 48
    private let maximumColumnWidth: CGFloat = 280

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let columnWidths = widths(for: subviews, availableWidth: proposal.width)
        let rowHeights = heights(for: subviews, columnWidths: columnWidths)

        return CGSize(width: columnWidths.reduce(0, +), height: rowHeights.reduce(0, +))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let columnWidths = widths(for: subviews, availableWidth: bounds.width)
        let rowHeights = heights(for: subviews, columnWidths: columnWidths)
        let rows = groupedSubviews(subviews)
        var currentY = bounds.minY

        for rowIndex in rows.keys.sorted() {
            var currentX = bounds.minX
            for columnIndex in 0..<columnCount {
                guard let subview = rows[rowIndex]?[columnIndex] else {
                    currentX += columnWidths[columnIndex]
                    continue
                }

                subview.place(
                    at: CGPoint(x: currentX, y: currentY),
                    anchor: .topLeading,
                    proposal: ProposedViewSize(width: columnWidths[columnIndex], height: rowHeights[rowIndex])
                )
                currentX += columnWidths[columnIndex]
            }
            currentY += rowHeights[rowIndex]
        }
    }

    private func widths(for subviews: Subviews, availableWidth: CGFloat?) -> [CGFloat] {
        guard columnCount > 0 else { return [] }

        let rows = groupedSubviews(subviews)
        let desiredWidths = (0..<columnCount).map { columnIndex in
            let measuredWidth = rows.values.compactMap { row in
                row[columnIndex]?.sizeThatFits(.unspecified).width
            }.max() ?? minimumColumnWidth

            return min(maximumColumnWidth, max(minimumColumnWidth, measuredWidth))
        }

        guard let availableWidth, availableWidth > 0 else { return desiredWidths }

        let desiredTotalWidth = desiredWidths.reduce(0, +)
        guard desiredTotalWidth > availableWidth else { return desiredWidths }

        let minimumWidths = desiredWidths.map { min($0, minimumColumnWidth) }
        let minimumTotalWidth = minimumWidths.reduce(0, +)
        guard minimumTotalWidth < availableWidth else { return minimumWidths }

        let flexibleWidths = zip(desiredWidths, minimumWidths).map(-)
        let totalFlexibleWidth = flexibleWidths.reduce(0, +)
        guard totalFlexibleWidth > 0 else { return minimumWidths }

        let remainingWidth = availableWidth - minimumTotalWidth
        return zip(minimumWidths, flexibleWidths).map { minimumWidth, flexibleWidth in
            minimumWidth + remainingWidth * (flexibleWidth / totalFlexibleWidth)
        }
    }

    private func heights(for subviews: Subviews, columnWidths: [CGFloat]) -> [CGFloat] {
        let rows = groupedSubviews(subviews)
        guard let maxRowIndex = rows.keys.max() else { return [] }

        return (0...maxRowIndex).map { rowIndex in
            (0..<columnCount).compactMap { columnIndex in
                rows[rowIndex]?[columnIndex]?.sizeThatFits(
                    ProposedViewSize(width: columnWidths[columnIndex], height: nil)
                ).height
            }.max() ?? 0
        }
    }

    private func groupedSubviews(_ subviews: Subviews) -> [Int: [Int: LayoutSubview]] {
        var rows: [Int: [Int: LayoutSubview]] = [:]
        for subview in subviews {
            rows[subview[ChatMarkdownTableRowKey.self], default: [:]][subview[ChatMarkdownTableColumnKey.self]] = subview
        }
        return rows
    }
}

private struct ChatMarkdownTableRowKey: LayoutValueKey {
    nonisolated static let defaultValue = 0
}

private struct ChatMarkdownTableColumnKey: LayoutValueKey {
    nonisolated static let defaultValue = 0
}

struct ChatReplyContent: Equatable {
    let body: String
    let author: String
    let summary: String
}

enum ChatReplyContentParser {
    static func parse(_ content: String) -> ChatReplyContent? {
        let separator = "\n\nIn reply to "
        guard let separatorRange = content.range(of: separator, options: .backwards) else { return nil }

        let body = String(content[..<separatorRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        let replyContext = String(content[separatorRange.upperBound...])
        guard let authorEndIndex = replyContext.firstIndex(of: ":") else { return nil }

        let author = String(replyContext[..<authorEndIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
        let quotedText = String(replyContext[replyContext.index(after: authorEndIndex)...])
        let summary = quotedText
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { line in
                var lineText = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if lineText.hasPrefix(">") {
                    lineText.removeFirst()
                }
                return lineText.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            .filter { !$0.isEmpty }
            .joined(separator: " ")

        guard !body.isEmpty, !author.isEmpty, !summary.isEmpty else { return nil }
        return ChatReplyContent(body: body, author: author, summary: summary)
    }
}

enum ChatMessageContentParser {
    /// Find-in-chat (#3045): every occurrence of `query` in `text` gets a
    /// solid background — yellow with primary (dark) text, readable on
    /// both bubble colours; the current match's message uses a stronger
    /// orange instead, for every occurrence in that message.
    static func highlighted(_ text: AttributedString, query: String, isCurrentMatch: Bool) -> AttributedString {
        var result = text
        let highlightColor: Color = isCurrentMatch ? .orange : .yellow
        var cursor = result.startIndex
        while cursor < result.endIndex,
              let range = result[cursor...].range(of: query, options: [.caseInsensitive]) {
            result[range].backgroundColor = highlightColor
            result[range].foregroundColor = .primary
            cursor = range.upperBound
        }
        return result
    }

    /// One flowing run of `ChatAttributedTextLine`'s inline layout: text
    /// (never split further — a highlighted or bold run embedded in plain
    /// text stays merged with its neighbours into one segment, so a
    /// multi-run `AttributedString` still wraps as a single paragraph),
    /// or a compact link, which needs its own tappable `Link` view and so
    /// can't be concatenated into the surrounding text.
    enum InlineSegment: Equatable {
        case text(AttributedString)
        case link(title: String, url: URL)
    }

    /// Splits `text` into the segments `ChatAttributedTextLine` renders:
    /// adjacent non-link runs merge into one `.text` segment. Before this,
    /// every attribute run (every highlighted word, every bold span) became
    /// its own `Text` subview in the custom flow layout — which places a
    /// whole subview on the current line or bumps it whole to the next,
    /// never splitting one subview's own text mid-word. That reads fine for
    /// a single run, but a run boundary that happens to fall near where the
    /// paragraph would naturally wrap forces a break right there and
    /// narrows the bubble to the shorter of the two pieces (#3045 — the
    /// same shape of break bold markdown text hit before this). Only an
    /// actual link still becomes its own segment — a `Text` and a `Link`
    /// can't merge into one view, so that boundary is real.
    static func inlineSegments(in text: AttributedString) -> [InlineSegment] {
        var segments: [InlineSegment] = []
        var pendingText: AttributedString?

        func flushPendingText() {
            guard let pending = pendingText else { return }
            segments.append(.text(pending))
            pendingText = nil
        }

        for run in text.runs {
            let runText = AttributedString(text[run.range])
            let displayText = String(runText.characters)

            if let url = run.link, url.host == displayText {
                flushPendingText()
                segments.append(.link(title: displayText, url: url))
            } else if pendingText != nil {
                pendingText?.append(runText)
            } else {
                pendingText = runText
            }
        }
        flushPendingText()

        return segments
    }

    static func parse(_ content: String) -> [ChatMessageContentBlock] {
        let lines = content.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var blocks: [ChatMessageContentBlock] = []
        var pendingTextLines: [String] = []
        var lineIndex = 0

        func flushText() {
            guard !pendingTextLines.isEmpty else { return }

            let text = pendingTextLines.joined(separator: "\n")
            blocks.append(ChatMessageContentBlock(id: blocks.count, kind: .text(markdownText(from: text))))
            pendingTextLines.removeAll()
        }

        while lineIndex < lines.count {
            if let table = table(startingAt: lineIndex, in: lines) {
                flushText()
                blocks.append(ChatMessageContentBlock(id: blocks.count, kind: .table(table.value)))
                lineIndex = table.endIndex
            } else {
                pendingTextLines.append(lines[lineIndex])
                lineIndex += 1
            }
        }

        flushText()

        if blocks.isEmpty {
            return [ChatMessageContentBlock(id: 0, kind: .text(AttributedString(" ")))]
        }

        return blocks
    }

    static func markdownText(from text: String) -> AttributedString {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var result = AttributedString()

        for (index, line) in lines.enumerated() {
            if index > 0 {
                result.append(AttributedString("\n"))
            }

            let lineText = line.isEmpty ? " " : line
            result.append(markdownLine(from: lineText))
        }

        return result
    }

    private static func markdownLine(from lineText: String) -> AttributedString {
        compactURLLikeLinkText(in: markdownAttributedString(from: markdownTextWithCompactedRawLinks(from: lineText)))
    }

    private static func markdownAttributedString(from text: String) -> AttributedString {
        guard let leadingEndIndex = text.firstIndex(where: { !$0.isWhitespace }) else {
            return AttributedString(text)
        }

        let trailingStartIndex = text.lastIndex(where: { !$0.isWhitespace }).map { text.index(after: $0) } ?? leadingEndIndex
        let markdownText = String(text[leadingEndIndex..<trailingStartIndex])
        var result = AttributedString(String(text[..<leadingEndIndex]))
        result.append((try? AttributedString(markdown: markdownText)) ?? AttributedString(markdownText))
        result.append(AttributedString(String(text[trailingStartIndex...])))
        return result
    }

    private static func markdownTextWithCompactedRawLinks(from lineText: String) -> String {
        let protectedRanges = markdownLinkSyntaxRanges(in: lineText)
        let links = rawLinks(in: lineText, excluding: protectedRanges)
        guard !links.isEmpty else { return lineText }

        var result = ""
        var currentIndex = lineText.startIndex

        for link in links {
            result += lineText[currentIndex..<link.range.lowerBound]
            result += "[\(link.displayText)](\(link.url.absoluteString))"
            currentIndex = link.range.upperBound
        }

        result += lineText[currentIndex...]
        return result
    }

    private static func compactURLLikeLinkText(in attributedText: AttributedString) -> AttributedString {
        var result = AttributedString()

        for run in attributedText.runs {
            guard let url = run.link, let host = url.host else {
                result.append(AttributedString(attributedText[run.range]))
                continue
            }

            let displayText = String(attributedText[run.range].characters)
            guard isURLLikeDisplayText(displayText) else {
                result.append(AttributedString(attributedText[run.range]))
                continue
            }

            var replacement = AttributedString(host)
            replacement.link = url
            result.append(replacement)
        }

        return result
    }

    private static func rawLinks(in lineText: String, excluding protectedRanges: [Range<String.Index>]) -> [RawLink] {
        var links: [RawLink] = []
        var index = lineText.startIndex

        while index < lineText.endIndex {
            if let protectedRange = protectedRanges.first(where: { $0.contains(index) }) {
                index = protectedRange.upperBound
                continue
            }

            guard startsRawLink(at: index, in: lineText) else {
                index = lineText.index(after: index)
                continue
            }

            let linkStartIndex = index
            let rangeStartIndex = previousCharacter(before: index, in: lineText) == "<" ? lineText.index(before: index) : index
            var scanIndex = index
            while scanIndex < lineText.endIndex, !isRawLinkTerminator(lineText[scanIndex]) {
                scanIndex = lineText.index(after: scanIndex)
            }

            var linkEndIndex = scanIndex
            while linkEndIndex > linkStartIndex, let lastCharacter = lineText[lineText.index(before: linkEndIndex)].unicodeScalars.first, CharacterSet(charactersIn: ".,!?;:)").contains(lastCharacter) {
                linkEndIndex = lineText.index(before: linkEndIndex)
            }

            let rangeEndIndex = scanIndex < lineText.endIndex && lineText[scanIndex] == ">" && rangeStartIndex < linkStartIndex ? lineText.index(after: scanIndex) : linkEndIndex
            let rawValue = String(lineText[linkStartIndex..<linkEndIndex])
            if linkEndIndex > linkStartIndex, let url = normalizedURL(from: rawValue), let host = url.host {
                links.append(RawLink(range: rangeStartIndex..<rangeEndIndex, url: url, displayText: host))
            }

            index = scanIndex
        }

        return links
    }

    private static func markdownLinkSyntaxRanges(in lineText: String) -> [Range<String.Index>] {
        var ranges: [Range<String.Index>] = []
        var searchIndex = lineText.startIndex

        while let labelStart = lineText[searchIndex...].firstIndex(of: "["),
              let labelEnd = lineText[labelStart...].firstIndex(of: "]") {
            let destinationStart = lineText.index(after: labelEnd)
            guard destinationStart < lineText.endIndex, lineText[destinationStart] == "(" else {
                searchIndex = lineText.index(after: labelStart)
                continue
            }

            let destinationContentStart = lineText.index(after: destinationStart)
            guard let destinationEnd = lineText[destinationContentStart...].firstIndex(of: ")") else { break }

            ranges.append(labelStart..<lineText.index(after: destinationEnd))
            searchIndex = lineText.index(after: destinationEnd)
        }

        return ranges
    }

    private static func startsRawLink(at index: String.Index, in lineText: String) -> Bool {
        let suffix = lineText[index...]
        return suffix.hasPrefix("https://") || suffix.hasPrefix("http://") || suffix.hasPrefix("www.")
    }

    private static func isRawLinkTerminator(_ character: Character) -> Bool {
        character.isWhitespace || character == "<" || character == ">" || character == "[" || character == "]"
    }

    private static func normalizedURL(from rawValue: String) -> URL? {
        if rawValue.hasPrefix("www.") {
            return URL(string: "https://\(rawValue)")
        }
        return URL(string: rawValue)
    }

    private static func previousCharacter(before index: String.Index, in text: String) -> Character? {
        guard index > text.startIndex else { return nil }
        return text[text.index(before: index)]
    }

    private static func isURLLikeDisplayText(_ displayText: String) -> Bool {
        let trimmedText = displayText.trimmingCharacters(in: .whitespacesAndNewlines)
        return normalizedURL(from: trimmedText)?.host != nil
    }

    private struct RawLink {
        let range: Range<String.Index>
        let url: URL
        let displayText: String
    }

    private static func table(startingAt index: Int, in lines: [String]) -> (value: ChatMarkdownTable, endIndex: Int)? {
        guard index + 1 < lines.count else { return nil }

        let headerLine = lines[index]
        let separatorLine = lines[index + 1]
        guard isPipeRow(headerLine), let alignments = parseSeparatorRow(separatorLine) else { return nil }

        let headers = parseRow(headerLine)
        guard !headers.isEmpty, alignments.count >= headers.count else { return nil }

        var rows: [[String]] = []
        var nextIndex = index + 2
        while nextIndex < lines.count {
            if isPipeRow(lines[nextIndex]) {
                let row = parseRow(lines[nextIndex])
                if row.count >= headers.count {
                    rows.append(row)
                } else if isTableRowContinuation(lines[nextIndex]), !rows.isEmpty {
                    appendContinuation(lines[nextIndex], toLastCellIn: &rows)
                } else if !row.isEmpty {
                    rows.append(row)
                }
                nextIndex += 1
            } else if isTableRowContinuation(lines[nextIndex]), !rows.isEmpty {
                appendContinuation(lines[nextIndex], toLastCellIn: &rows)
                nextIndex += 1
            } else {
                break
            }
        }

        guard !rows.isEmpty else { return nil }

        return (ChatMarkdownTable(headers: headers, alignments: alignments, rows: rows), nextIndex)
    }

    private static func isPipeRow(_ line: String) -> Bool {
        let trimmedLine = line.trimmingCharacters(in: .whitespaces)
        return trimmedLine.contains("|") && !trimmedLine.isEmpty
    }

    private static func isTableRowContinuation(_ line: String) -> Bool {
        guard let firstCharacter = line.first, firstCharacter.isWhitespace else { return false }
        return !line.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private static func appendContinuation(_ line: String, toLastCellIn rows: inout [[String]]) {
        guard let lastRowIndex = rows.indices.last, let lastCellIndex = rows[lastRowIndex].indices.last else { return }

        var continuation = line.trimmingCharacters(in: .whitespaces)
        if continuation.last == "|" {
            continuation.removeLast()
            continuation = continuation.trimmingCharacters(in: .whitespaces)
        }
        rows[lastRowIndex][lastCellIndex] += "\n" + continuation
    }

    private static func parseRow(_ line: String) -> [String] {
        let row = String(trimOuterPipes(line))
        var cells: [String] = []
        var currentCell = ""
        var index = row.startIndex
        var isEscaped = false
        var isInsideCodeSpan = false
        var activeEmphasisDelimiter: Character?

        while index < row.endIndex {
            let character = row[index]

            if isEscaped {
                currentCell.append(character)
                isEscaped = false
                index = row.index(after: index)
                continue
            }

            if character == "\\" {
                isEscaped = true
                index = row.index(after: index)
                continue
            }

            if character == "`" {
                isInsideCodeSpan.toggle()
                currentCell.append(character)
                index = row.index(after: index)
                continue
            }

            if !isInsideCodeSpan, character == "*" || character == "_" {
                let delimiter = character
                let runStart = index
                repeat {
                    index = row.index(after: index)
                } while index < row.endIndex && row[index] == delimiter

                if activeEmphasisDelimiter == delimiter {
                    activeEmphasisDelimiter = nil
                } else if activeEmphasisDelimiter == nil {
                    activeEmphasisDelimiter = delimiter
                }

                currentCell.append(contentsOf: row[runStart..<index])
                continue
            }

            if character == "|", !isInsideCodeSpan, activeEmphasisDelimiter == nil {
                cells.append(currentCell.trimmingCharacters(in: .whitespaces))
                currentCell.removeAll()
            } else {
                currentCell.append(character)
            }

            index = row.index(after: index)
        }

        if isEscaped {
            currentCell.append("\\")
        }

        cells.append(currentCell.trimmingCharacters(in: .whitespaces))
        return cells
    }

    private static func parseSeparatorRow(_ line: String) -> [ChatMarkdownTable.ColumnAlignment]? {
        guard isPipeRow(line) else { return nil }

        let cells = parseRow(line)
        guard !cells.isEmpty else { return nil }

        var alignments: [ChatMarkdownTable.ColumnAlignment] = []
        for cell in cells {
            guard let alignment = parseSeparatorCell(cell) else { return nil }
            alignments.append(alignment)
        }

        return alignments
    }

    private static func parseSeparatorCell(_ cell: String) -> ChatMarkdownTable.ColumnAlignment? {
        let trimmedCell = cell.trimmingCharacters(in: .whitespaces)
        guard trimmedCell.count >= 3 else { return nil }

        let hasLeadingColon = trimmedCell.hasPrefix(":")
        let hasTrailingColon = trimmedCell.hasSuffix(":")
        let dashStartIndex = hasLeadingColon ? trimmedCell.index(after: trimmedCell.startIndex) : trimmedCell.startIndex
        let dashEndIndex = hasTrailingColon ? trimmedCell.index(before: trimmedCell.endIndex) : trimmedCell.endIndex
        guard dashStartIndex < dashEndIndex else { return nil }

        let dashes = trimmedCell[dashStartIndex..<dashEndIndex]
        guard dashes.allSatisfy({ $0 == "-" }), dashes.count >= 3 else { return nil }

        if hasLeadingColon && hasTrailingColon {
            return .center
        }

        if hasTrailingColon {
            return .trailing
        }

        return .leading
    }

    private static func trimOuterPipes(_ line: String) -> Substring {
        var trimmedLine = line.trimmingCharacters(in: .whitespaces)[...]

        if trimmedLine.first == "|" {
            trimmedLine.removeFirst()
        }

        if trimmedLine.last == "|" {
            trimmedLine.removeLast()
        }

        return trimmedLine
    }
}

extension MessageView.MessageLayout {
    var columnCount: Int {
        switch self {
        case .message:
            return 1
        case .compact:
            return 2
        case .media:
            return 3
        }
    }

    var horizontalPadding: CGFloat {
        switch self {
        case .message:
            return 16
        case .compact:
            return 10
        case .media:
            return 8
        }
    }

    var itemSpacing: CGFloat {
        switch self {
        case .message:
            return 12
        case .compact:
            return 8
        case .media:
            return 6
        }
    }
}
