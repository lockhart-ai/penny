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

    @Test func preservesEmptyLinesAroundCompactLinks() {
        let line = "https://store.example.com."
        let title = "**Extra Perks:** Bonus content."
        let lineHeight = renderedHeight(" ", width: 360)
        let content = "\n\(line)\n\n\(title)\n"
        let expectedHeight = renderedHeight(line, width: 360) + renderedHeight(title, width: 360) + 3 * lineHeight

        #expect(abs(renderedHeight(content, width: 360) - expectedHeight) < 1)
    }

    private func renderedHeight(_ markdown: String, width: CGFloat, textSize: DynamicTypeSize = .large) -> CGFloat {
        let content = ChatAttributedText(attributedText: ChatMessageContentParser.markdownText(from: markdown))
            .font(.body)
            .environment(\.dynamicTypeSize, textSize)
        let host = UIHostingController(rootView: content)
        return host.sizeThatFits(in: CGSize(width: width, height: .greatestFiniteMagnitude)).height
    }
}
