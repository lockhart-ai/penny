import Foundation
import SwiftUI
import Testing
import UIKit
@testable import PennyClient

@MainActor
struct ChatAttributedTextTests {
    // The reported link/title boundaries, with synthetic URLs and message details.
    private static let examples = [
        """
        Key takeaways:
        - **Release Date:** Confirmed for **next month** https://news.example.com, https://store.example.com/details.
        - **Platforms:** Console and PC https://news.example.com, https://store.example.com.
        - **Setting:** A coastal city https://world.example.com.
        - **Engine/Tech:** Updated engine and NPC dialogue https://tech.example.com.
        - **Pre-order info:** Official details are set https://store.example.com.
        """,
        """
        **2. Bonuses and Editions**
        *   **Standard vs. Ultimate:** The **Ultimate Edition** is priced at **$99.99** https://store.example.com/preorder.
        *   **Extra Perks:** Look for pre-order bonuses https://store.example.com/preorder, https://bonuses.example.com.

        **3. Convenience & Payment**
        *   **When you pay:** Check the payment date https://store.example.com/preorder.
        """
    ]

    @Test(arguments: [CGFloat(280), 360, 600], [DynamicTypeSize.large, .accessibility3])
    func sourceLinesKeepSeparateRows(width: CGFloat, textSize: DynamicTypeSize) {
        for content in Self.examples {
            let expectedHeight = content.components(separatedBy: "\n").reduce(CGFloat.zero) { height, line in
                height + renderedHeight(line.isEmpty ? " " : line, width: width, textSize: textSize)
            }

            let actualHeight = renderedHeight(content, width: width, textSize: textSize)

            #expect(abs(actualHeight - expectedHeight) < 1)
        }
    }

    @Test(arguments: [CGFloat(280), 360], [DynamicTypeSize.large, .accessibility3])
    func longParagraphWrapsToItsFullHeight(width: CGFloat, textSize: DynamicTypeSize) {
        let paragraph = Array(repeating: "A detailed collection update should wrap and remain fully readable.", count: 8).joined(separator: " ")
        let nativeText = Text(paragraph)
            .font(.body)
            .fixedSize(horizontal: false, vertical: true)
            .environment(\.dynamicTypeSize, textSize)
        let host = UIHostingController(rootView: nativeText)
        let expectedHeight = host.sizeThatFits(in: CGSize(width: width, height: .greatestFiniteMagnitude)).height

        #expect(renderedHeight(paragraph, width: width, textSize: textSize) >= expectedHeight - 1)
    }

    /// Find-in-chat (#3045): highlighting one word partway through a
    /// sentence must only recolour that run, never split the sentence onto
    /// its own forced line — the regression this guards is a break right
    /// after the highlighted word, with the bubble narrowing to match
    /// (`ChatAttributedTextLine` turning each attribute run into its own
    /// flow-layout block instead of keeping the paragraph's runs merged).
    @Test(arguments: [CGFloat(200), 280, 360], [DynamicTypeSize.large, .accessibility3])
    func highlightingAMidSentenceWordDoesNotAddALineBreak(width: CGFloat, textSize: DynamicTypeSize) {
        let sentence = "The 9:30 ferry has space — want me to book it?"
        let plainHeight = renderedHeight(sentence, width: width, textSize: textSize)

        let highlightedText = ChatMessageContentParser.highlighted(
            ChatMessageContentParser.markdownText(from: sentence), query: "ferry", isCurrentMatch: false
        )
        let highlightedHeight = renderedHeight(highlightedText, width: width, textSize: textSize)

        #expect(abs(highlightedHeight - plainHeight) < 1)
    }

    @Test func preservesEmptyLinesAroundCompactLinks() {
        let line = "https://store.example.com."
        let title = "**Extra Perks:** Bonus content."
        let lineHeight = renderedHeight(" ", width: 360)
        let content = "\n\(line)\n\n\(title)\n"
        let expectedHeight = renderedHeight(line, width: 360) + renderedHeight(title, width: 360) + 3 * lineHeight

        #expect(abs(renderedHeight(content, width: 360) - expectedHeight) < 1)
    }

    private func renderedHeight(_ markdown: String, width: CGFloat, textSize: DynamicTypeSize = .large) -> CGFloat {
        renderedHeight(ChatMessageContentParser.markdownText(from: markdown), width: width, textSize: textSize)
    }

    private func renderedHeight(_ attributedText: AttributedString, width: CGFloat, textSize: DynamicTypeSize = .large) -> CGFloat {
        let content = ChatAttributedText(attributedText: attributedText)
            .font(.body)
            .environment(\.dynamicTypeSize, textSize)
        let host = UIHostingController(rootView: content)
        return host.sizeThatFits(in: CGSize(width: width, height: .greatestFiniteMagnitude)).height
    }
}
