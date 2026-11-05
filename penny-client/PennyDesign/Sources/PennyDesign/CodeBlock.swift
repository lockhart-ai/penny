import SwiftUI
import UIKit

/// Raw machine text on the read pages — the system prompt, a thinking
/// block, a tool call's arguments and result. Read-only, with basic
/// styling cues only, not an editor: a line-number gutter on the left
/// (caption2 monospaced digits, tertiary, right-aligned, fixed to the
/// width of the largest number) and the text in footnote monospaced,
/// long lines wrapping on a hanging indent — the continuation aligns
/// with the line's own indentation, and only the line's first visual row
/// carries a number. JSON is coloured by a simple tokeniser (keys blue,
/// strings red, numbers purple, true/false/null pink, punctuation
/// secondary); markdown-ish prose renders `##` headings bold and list
/// markers secondary, the rest primary. Selectable; the caller puts it
/// inside the shared white card.
public struct CodeBlock: View {
    public enum Mode {
        /// Pretty-printed JSON, 2-space indentation.
        case json
        /// Markdown-ish prose (system prompts, thinking).
        case markdown
    }

    public let text: String
    public var mode: Mode = .markdown

    public init(text: String, mode: Mode = .markdown) {
        self.text = text
        self.mode = mode
    }

    public var body: some View {
        let rows = text.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                CodeLine(number: index + 1, text: row, mode: mode,
                         digits: max(String(rows.count).count, 1))
            }
        }
        .textSelection(.enabled)
    }
}

/// One source line: the gutter number on the first (and only) visual row,
/// the line's text after its own indentation — which the wrapped
/// continuation shares, because the indent rides the text frame's leading
/// padding, not the first line.
private struct CodeLine: View {
    let number: Int
    let text: String
    let mode: CodeBlock.Mode
    /// Digits in the largest line number, so every gutter on the page is
    /// the same width and the numbers right-align inside it.
    let digits: Int

    /// The footnote monospaced font the text renders in, and one of its
    /// characters' width — the unit of the hanging indent.
    static let monoFont: UIFont = {
        let base = uiFont(.footnote, .regular)
        return UIFont(descriptor: base.fontDescriptor.withDesign(.monospaced)
            ?? base.fontDescriptor, size: base.pointSize)
    }()
    static let charWidth: CGFloat = ("0" as NSString)
        .size(withAttributes: [.font: monoFont]).width
    /// The gutter's monospaced-digit font (caption2) and its digit width.
    static let gutterFont: UIFont = .monospacedDigitSystemFont(
        ofSize: uiFont(.caption2, .regular).pointSize, weight: .regular)
    static let gutterDigitWidth: CGFloat = ("0" as NSString)
        .size(withAttributes: [.font: gutterFont]).width

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Tokens.iconTextGap) {
            Text("\(number)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
                .frame(width: CGFloat(digits) * Self.gutterDigitWidth + 1,
                       alignment: .trailing)
            line
                .font(.footnote.monospaced())
                .padding(.leading, CGFloat(leadingSpaces) * Self.charWidth)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var leadingSpaces: Int {
        text.prefix(while: { $0 == " " }).count
    }

    private var line: Text {
        let body = String(text.dropFirst(leadingSpaces))
        if body.isEmpty { return Text(" ") }
        switch mode {
        case .json: return Self.jsonLine(body)
        case .markdown: return Self.markdownLine(body)
        }
    }

    // MARK: JSON

    /// One JSON line as coloured fragments: strings, numbers and literals
    /// tokenised; everything between tokens (whitespace, braces, brackets,
    /// commas, colons) secondary.
    private static func jsonLine(_ body: String) -> Text {
        guard let tokens = jsonTokens else {
            // Unreachable with the constant pattern above; render plain.
            return Text(AttributedString(body))
        }
        var pieces: [(text: String, color: Color)] = []
        var cursor = body.startIndex
        let nsBody = body as NSString
        for match in tokens.matches(in: body, range: NSRange(location: 0, length: nsBody.length)) {
            let range = Range(match.range, in: body)!
            if range.lowerBound > cursor {
                pieces.append((String(body[cursor..<range.lowerBound]), .secondary))
            }
            let token = String(body[range])
            pieces.append((token, jsonColor(token, after: range.upperBound, in: body)))
            cursor = range.upperBound
        }
        if cursor < body.endIndex {
            pieces.append((String(body[cursor...]), .secondary))
        }
        return colouredLine(pieces)
    }

    /// Strings, numbers, literals and single punctuation characters. The
    /// pattern is a compile-time constant and always parses; the optional
    /// exists only to keep `try!` out of the kit.
    private static let jsonTokens = try? NSRegularExpression(
        pattern: #""(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[{}\[\],:]"#)

    /// A string followed (after spaces) by a colon is a key; the literals
    /// are pink; anything starting like a number is purple.
    private static func jsonColor(_ token: String, after end: String.Index,
                                  in body: String) -> Color {
        if token.hasPrefix("\"") {
            var probe = end
            while probe < body.endIndex, body[probe] == " " { probe = body.index(after: probe) }
            return probe < body.endIndex && body[probe] == ":" ? .blue : .red
        }
        switch token {
        case "true", "false", "null": return .pink
        default:
            return token.first.map { $0.isNumber || $0 == "-" } == true ? .purple : .secondary
        }
    }

    // MARK: Shared rendering

    /// Renders the fragments as one line of attributed text. (The `+`
    /// operator on `Text` is deprecated on iOS 26; attributed strings are
    /// the way to colour one line, and render identically here.)
    private static func colouredLine(_ pieces: [(text: String, color: Color)]) -> Text {
        var out = AttributedString()
        for piece in pieces {
            var run = AttributedString(piece.text)
            run.foregroundColor = piece.color
            out += run
        }
        return Text(out)
    }

    // MARK: Markdown-ish prose

    private static func markdownLine(_ body: String) -> Text {
        if body.hasPrefix("#") {
            var run = AttributedString(body)
            run.inlinePresentationIntent = .stronglyEmphasized
            return Text(run)
        }
        if let marker = body.range(of: #"^(- |\d+\. )"#, options: .regularExpression) {
            return colouredLine([
                (String(body[marker]), .secondary),
                (String(body[marker.upperBound...]), .primary)
            ])
        }
        var run = AttributedString(body)
        run.foregroundColor = .primary
        return Text(run)
    }
}
