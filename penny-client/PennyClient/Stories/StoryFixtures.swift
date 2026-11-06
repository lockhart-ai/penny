#if DEBUG
import Foundation
import PennyServices

/// Contrived fixture data for story mode (#2965) — never real user content.
/// One factory builds every chat story's `MessageView.ViewModel`; the data
/// browser stories reuse `DataBrowserFixtures` directly (see
/// `DataBrowserPreviews.swift`), rather than a second fixture set.
@MainActor
enum StoryFixtures {
    /// A tiny, valid solid-color 8x8 PNG, embedded as a data URL — a plain
    /// placeholder that reads as a photo, not a broken image, for the one
    /// story that needs an image attachment.
    private static let tinyImageDataURL =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEUlEQVR42mM4MisHK2IYWhIA01xygQpKgBgAAAAASUVORK5CYII="

    static func chatViewModel(
        messages: [ChatMessage],
        isConnected: Bool = true,
        workingSteps: [StoryWorkingStep] = [],
        backgroundRunCount: Int = 0
    ) -> MessageView.ViewModel {
        let client = PennyService.storyInstance(
            messages: messages,
            isConnected: isConnected,
            workingSteps: workingSteps,
            backgroundRunCount: backgroundRunCount
        )
        return MessageView.ViewModel(client: client)
    }

    static func workingViewModel() -> MessageView.ViewModel {
        chatViewModel(messages: conversationMessages, workingSteps: workingSteps)
    }

    static func disconnectedViewModel() -> MessageView.ViewModel {
        chatViewModel(messages: conversationMessages, isConnected: false)
    }

    /// Two collectors running in the background — the Prompts tab badge
    /// story (#3046).
    static func collectorsBadgeViewModel() -> MessageView.ViewModel {
        chatViewModel(messages: conversationMessages, backgroundRunCount: 2)
    }

    /// Three steps with different tools, the last one unfinished — the run
    /// "Penny is working" stories show as still in progress.
    static var workingSteps: [StoryWorkingStep] {
        [
            StoryWorkingStep(toolName: "browse", arguments: ["queries": .array([.string("moka pot coffee bean grind size")])]),
            StoryWorkingStep(toolName: "memory_lookup", arguments: ["query": .string("coffee bean grind size notes")]),
            StoryWorkingStep(toolName: "browse", arguments: ["queries": .array([.string("https://example.com/coffee-roasting-guide")])])
        ]
    }

    /// A mixed conversation about brewing coffee: markdown emphasis, a link,
    /// a table, an image attachment, and a reply/quote.
    static var conversationMessages: [ChatMessage] {
        let now = Date()
        func at(_ secondsAgo: TimeInterval) -> Date { now.addingTimeInterval(-secondsAgo) }

        return [
            ChatMessage(
                id: 1, serverID: 1, createdAt: at(300),
                content: "What roast works best for a moka pot?",
                sourceHint: nil, imageAttachments: [], isOutgoing: true
            ),
            ChatMessage(
                id: 2, serverID: 2, createdAt: at(280),
                content: "A **medium-dark** roast with a fine, even grind usually works best for stovetop brewing. "
                    + "Found a write-up here: [roasting guide](https://example.com/coffee-roasting-guide).",
                sourceHint: nil, imageAttachments: [], isOutgoing: false
            ),
            ChatMessage(
                id: 3, serverID: 3, createdAt: at(220),
                content: "What about grind size and price for a hand grinder?",
                sourceHint: nil, imageAttachments: [], isOutgoing: true
            ),
            ChatMessage(
                id: 4, serverID: 4, createdAt: at(200),
                content: """
                | Grinder | Grind range | Price |
                | --- | --- | --- |
                | Basic hand grinder | Coarse–fine | $35 |
                | Comandante C40 | Coarse–fine | $220 |
                """,
                sourceHint: nil, imageAttachments: [], isOutgoing: false
            ),
            ChatMessage(
                id: 5, serverID: 5, createdAt: at(180),
                content: "Here's a grind-size reference chart.",
                sourceHint: nil, imageAttachmentDataURLs: [tinyImageDataURL], imageAttachments: [], isOutgoing: false
            ),
            ChatMessage(
                id: 6, serverID: 6, createdAt: at(60),
                content: """
                Can you also check if it's on sale this weekend?

                In reply to Penny:
                > A medium-dark roast with a fine, even grind usually works best for stovetop brewing.
                """,
                sourceHint: nil, parentID: 2, imageAttachments: [], isOutgoing: true
            )
        ]
    }
}
#endif
