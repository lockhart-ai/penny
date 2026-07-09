import Foundation
import SwiftUI

struct ChatMessageView: View {
    let message: ChatMessage
    let layout: MessageView.MessageLayout
    let showsSourceHintInline: Bool
    let fillsMessageRowWidth: Bool

    init(message: ChatMessage, layout: MessageView.MessageLayout, showsSourceHintInline: Bool = true, fillsMessageRowWidth: Bool = false) {
        self.message = message
        self.layout = layout
        self.showsSourceHintInline = showsSourceHintInline
        self.fillsMessageRowWidth = fillsMessageRowWidth
    }

    private var messageContentBlocks: [ChatMessageContentBlock] {
        ChatMessageContentParser.parse(message.content)
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

    @ViewBuilder
    private var messageBubble: some View {
        if message.isOutgoing {
            Text(message.content)
                .font(.body)
                .foregroundStyle(.white)
                .frame(maxWidth: fillsMessageRowWidth ? .infinity : nil, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        } else {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(messageContentBlocks) { block in
                    messageContentBlock(block)
                }

                ForEach(message.imageAttachments) { attachment in
                    Image(uiImage: attachment.image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: 260)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
            }
            .foregroundStyle(.primary)
            .frame(maxWidth: fillsMessageRowWidth ? .infinity : nil, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
    }

    @ViewBuilder
    private func messageContentBlock(_ block: ChatMessageContentBlock) -> some View {
        switch block.kind {
        case .text(let text):
            ChatAttributedText(attributedText: text)
                .lineLimit(nil)
                .font(.body)
        case .table(let table):
            markdownTable(table)
        }
    }

    private func markdownTable(_ table: ChatMarkdownTable) -> some View {
        ChatMarkdownTableView(table: table)
    }

    var body: some View {
        switch layout {
        case .message:
            messageRow
        case .compact:
            compactCard
        case .media:
            mediaCard
        }
    }

    private var messageRow: some View {
        HStack {
            if message.isOutgoing && !fillsMessageRowWidth {
                Spacer(minLength: 48)
            }

            VStack(alignment: message.isOutgoing && !fillsMessageRowWidth ? .trailing : .leading, spacing: 5) {
                if showsSourceHintInline, let sourceHint = message.sourceHint, !sourceHint.isEmpty {
                    Text(sourceHint)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }

                messageBubble

                Text(message.displayTime)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if !message.isOutgoing && !fillsMessageRowWidth {
                Spacer(minLength: 48)
            }
        }
        .frame(maxWidth: .infinity)
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

private struct ChatAttributedText: View {
    let attributedText: AttributedString

    var body: some View {
        ChatInlineFlowLayout(horizontalSpacing: 0, verticalSpacing: 0) {
            ForEach(fragments) { fragment in
                switch fragment.kind {
                case .text(let text):
                    Text(text)
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
        var fragments: [Fragment] = []
        var id = 0

        for run in attributedText.runs {
            let runText = AttributedString(attributedText[run.range])
            let displayText = String(runText.characters)

            if let url = run.link, url.host == displayText {
                fragments.append(Fragment(id: id, kind: .compactLink(title: displayText, url: url)))
            } else {
                fragments.append(Fragment(id: id, kind: .text(runText)))
            }

            id += 1
        }

        return fragments
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

enum ChatMessageContentParser {
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
