#if DEBUG
import Foundation
import UIKit
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

    /// Another tiny solid-color PNG for the bubble story (#3038): a
    /// landscape 8x5, a flair photo's 16:10 shape.
    private static let tinyLandscapeImageDataURL =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAIAAAD38zoCAAAAEUlEQVR4nGPQz5yIFTHQQQIARjYuaXcnV0MAAAAASUVORK5CYII="

    static func chatViewModel(
        messages: [ChatMessage],
        isConnected: Bool = true,
        workingSteps: [StoryWorkingStep] = [],
        workingElapsedSeconds: Int = 0,
        backgroundRunCount: Int = 0
    ) -> MessageView.ViewModel {
        let client = PennyService.storyInstance(
            messages: messages,
            isConnected: isConnected,
            workingSteps: workingSteps,
            workingElapsedSeconds: workingElapsedSeconds,
            backgroundRunCount: backgroundRunCount
        )
        return MessageView.ViewModel(client: client)
    }

    /// The live dock just started (#3042, `chat-working-start`): one step,
    /// its thinking already in but no tool call yet — the dock's very first
    /// visible frame.
    static func workingStartViewModel() -> MessageView.ViewModel {
        chatViewModel(
            messages: conversationMessages,
            workingSteps: [StoryWorkingStep(thinking: "Checking what roasts are back in stock this week.")],
            workingElapsedSeconds: 2
        )
    }

    /// The live dock mid-turn (#3042, `chat-working`): three steps, each
    /// with its own thinking line before its tool row, the last still
    /// running.
    static func workingViewModel() -> MessageView.ViewModel {
        chatViewModel(messages: conversationMessages, workingSteps: workingSteps, workingElapsedSeconds: 8)
    }

    /// The live dock at its 40%-of-screen cap (#3042, `chat-working-long`):
    /// enough steps that the stream scrolls internally, pinned to the
    /// newest item, with the top fade showing.
    static func workingLongViewModel() -> MessageView.ViewModel {
        chatViewModel(messages: conversationMessages, workingSteps: longWorkingSteps, workingElapsedSeconds: 22)
    }

    /// The live dock with a pending permission row (#3044, `chat-permission`):
    /// the same mid-turn steps as `workingViewModel()`, plus a pending
    /// `permissionPrompt` — the dock's trailing row asking to read a domain,
    /// with Deny and Allow.
    static func workingWithPermissionViewModel() -> MessageView.ViewModel {
        let client = PennyService.storyInstance(
            messages: conversationMessages,
            workingSteps: workingSteps,
            workingElapsedSeconds: 8,
            pendingPermissionDomain: "example.com"
        )
        return MessageView.ViewModel(client: client)
    }

    /// The dock just after a turn ends (#3042, #3130, `chat-resolving`):
    /// drives the real state machine rather than freezing a hand-picked
    /// blend. `syncDock()` shows the working run; the fixture then ends it
    /// (removing it from `agentProgressRuns`, the same way a real
    /// `run_finished` does) and `beginResolvingIfMatched` marks the
    /// already-arrived reply resolved — but with no `MessageView` mounted
    /// yet to drive the real `withAnimation(...) completion:`, nothing
    /// here finishes the exit. `MessageView`'s own `.onAppear` syncs the
    /// dock unconditionally (no story-only branch left, #3130) and finds
    /// no live run, so `handleForegroundRunEnded`'s stuck-`.resolving`
    /// recovery finishes it immediately on mount — the same fallback a
    /// real backgrounded/tab-switched-away dock relies on. The shot lands
    /// on the settled state — the dock gone, the reply in place — never an
    /// unreproducible mid-flight frame.
    static func resolvingViewModel() -> MessageView.ViewModel {
        let viewModel = chatViewModel(
            messages: bubbleMessages + [resolvingUserQuestion, resolvingReplyMessage],
            workingSteps: workingSteps,
            workingElapsedSeconds: 8
        )
        viewModel.syncDock()
        viewModel.client.agentProgressRuns.removeValue(forKey: "story-run")
        viewModel.beginResolvingIfMatched(appendedMessages: [resolvingReplyMessage])
        return viewModel
    }

    /// One message sent while she's working (#3043, `chat-queued`): the
    /// normal blue bubble at reduced opacity, "🕒 Queued" standing in for
    /// its time, above the live dock. `queuedMessageIDs` set directly —
    /// same shortcut every other story uses into `ViewModel`'s own state.
    static func queuedViewModel() -> MessageView.ViewModel {
        let queued = queuedMessages(count: 1)
        let viewModel = chatViewModel(
            messages: conversationMessages + queued,
            workingSteps: workingSteps,
            workingElapsedSeconds: 8
        )
        markQueued(queued, on: viewModel)
        return viewModel
    }

    /// Two messages sent while she's working (#3043, `chat-queued-two`) —
    /// the dock header's "· 2 queued".
    static func queuedTwoViewModel() -> MessageView.ViewModel {
        let queued = queuedMessages(count: 2)
        let viewModel = chatViewModel(
            messages: conversationMessages + queued,
            workingSteps: workingSteps,
            workingElapsedSeconds: 8
        )
        markQueued(queued, on: viewModel)
        return viewModel
    }

    /// Marks each message queued (#3043), with its step baseline set to the
    /// run's own step count (`workingSteps.count`) — matching a real send,
    /// where `sendDraft()` always records both together. Leaving the
    /// baseline at its default of 0 would read as "sent before step 1",
    /// so the view's own `syncDock()` clears the mark again on `.onAppear`,
    /// before the story ever renders it.
    private static func markQueued(_ queued: [ChatMessage], on viewModel: MessageView.ViewModel) {
        viewModel.queuedMessageIDs = Set(queued.map(\.id))
        for message in queued {
            viewModel.queuedMessageStepBaselines[message.id] = workingSteps.count
        }
    }

    private static func queuedMessages(count: Int) -> [ChatMessage] {
        let texts = ["Oh, and grab a bag of the dark roast too?", "Actually the decaf, if they have it."]
        return (0..<count).map { index in
            ChatMessage.local(id: 9_001 + index, content: texts[index])
        }
    }

    static func disconnectedViewModel() -> MessageView.ViewModel {
        chatViewModel(messages: conversationMessages, isConnected: false)
    }

    /// Two collectors running in the background — the Prompts tab badge
    /// story (#3046).
    static func collectorsBadgeViewModel() -> MessageView.ViewModel {
        chatViewModel(messages: conversationMessages, backgroundRunCount: 2)
    }

    /// The composer typing story (#3039): a draft with no photo — the send
    /// arrow shows inside the field, the `+` beside it.
    static func composerTypingViewModel() -> MessageView.ViewModel {
        let viewModel = chatViewModel(messages: conversationMessages)
        viewModel.draftMessage = "Perfect. And what's the weather for the crossing?"
        return viewModel
    }

    /// The composer photo-pending story (#3039): the `chat-pending` screen —
    /// a photo's thumbnail above the text line in the field, send arrow on.
    static func composerPhotoViewModel() -> MessageView.ViewModel {
        let viewModel = chatViewModel(messages: conversationMessages)
        viewModel.draftMessage = "What's this bird?"
        viewModel.pendingImage = pendingPhoto
        return viewModel
    }

    /// Find-in-chat open with matches (#3045): "ferry" appears in all three
    /// plain-text bubbles — both bubble colours, no table or reply-quote
    /// involved (those don't highlight yet — see `ChatMessageView`'s known
    /// gap), so the story shows a clean, readable highlight.
    static func chatSearchViewModel(query: String = "ferry") -> MessageView.ViewModel {
        let viewModel = chatViewModel(messages: findInChatMessages)
        viewModel.search.open()
        viewModel.search.query = query
        return viewModel
    }

    /// Find-in-chat open with an empty field (#3045): the placeholder, no
    /// highlights, the step arrows dimmed.
    static func chatSearchEmptyViewModel() -> MessageView.ViewModel {
        let viewModel = chatViewModel(messages: findInChatMessages)
        viewModel.search.open()
        return viewModel
    }

    /// A rendered placeholder photo for the pending state — a gradient sky
    /// over a stylised bird on a branch, so the composer's thumbnail reads
    /// as a photo, not a grey block. Rendered here, since the composer's
    /// pending image is a `UIImage`, not a data URL like the bubbles'
    /// attachments.
    private static var pendingPhoto: UIImage {
        let size = CGSize(width: 800, height: 500)
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { context in
            context.cgContext.drawLinearGradient(
                CGGradient(
                    colorsSpace: CGColorSpaceCreateDeviceRGB(),
                    colors: [
                        UIColor(red: 0.98, green: 0.84, blue: 0.62, alpha: 1).cgColor,
                        UIColor(red: 0.55, green: 0.72, blue: 0.83, alpha: 1).cgColor
                    ] as CFArray,
                    locations: [0, 1]
                )!,
                start: .zero,
                end: CGPoint(x: 0, y: size.height),
                options: []
            )
            // A branch across the lower third.
            UIColor(red: 0.36, green: 0.25, blue: 0.17, alpha: 1).setFill()
            context.fill(CGRect(x: 60, y: 360, width: 700, height: 18))
            // The bird: a robin's breast and wing, dark head, on the branch.
            UIColor(red: 0.35, green: 0.55, blue: 0.75, alpha: 1).setFill()
            context.fill(CGRect(x: 320, y: 250, width: 170, height: 120))
            UIColor(red: 0.86, green: 0.48, blue: 0.28, alpha: 1).setFill()
            context.fill(CGRect(x: 330, y: 300, width: 150, height: 70))
            UIColor(red: 0.16, green: 0.20, blue: 0.26, alpha: 1).setFill()
            context.fill(CGRect(x: 470, y: 240, width: 70, height: 60))
            // A tail.
            context.fill(CGRect(x: 280, y: 270, width: 50, height: 40))
        }
    }

    /// Three steps with different tools, the last one unfinished — the run
    /// "Penny is working" stories show as still in progress. Each carries
    /// its own thinking line, the live dock's stream shape (#3042).
    static var workingSteps: [StoryWorkingStep] {
        [
            StoryWorkingStep(
                toolName: "browse",
                arguments: ["queries": .array([.string("moka pot coffee bean grind size")])],
                thinking: "Checking what grind size works best for a moka pot."
            ),
            StoryWorkingStep(
                toolName: "memory_lookup",
                arguments: ["query": .string("coffee bean grind size notes")],
                thinking: "Worth checking what she's already noted about grind size."
            ),
            StoryWorkingStep(
                toolName: "browse",
                arguments: ["queries": .array([.string("https://example.com/coffee-roasting-guide")])],
                thinking: "That roasting guide should have a sizing chart."
            )
        ]
    }

    /// Enough steps to push the live dock past its 40%-of-screen cap
    /// (#3042, `chat-working-long`) — the internal scroll and the top fade.
    static var longWorkingSteps: [StoryWorkingStep] {
        [
            StoryWorkingStep(toolName: "browse", arguments: ["queries": .array([.string("moka pot coffee bean grind size")])], thinking: "Checking what grind size works best for a moka pot."),
            StoryWorkingStep(toolName: "memory_lookup", arguments: ["query": .string("coffee bean grind size notes")], thinking: "Worth checking what she's already noted about grind size."),
            StoryWorkingStep(toolName: "browse", arguments: ["queries": .array([.string("https://example.com/coffee-roasting-guide")])], thinking: "That roasting guide should have a sizing chart."),
            StoryWorkingStep(toolName: "memory_lookup", arguments: ["query": .string("hand grinder price notes")], thinking: "They asked about grinder prices too — checking memory first."),
            StoryWorkingStep(toolName: "browse", arguments: ["queries": .array([.string("comandante c40 price")])], thinking: "Worth a quick price check on the Comandante."),
            StoryWorkingStep(toolName: "browse", arguments: ["queries": .array([.string("https://example.com/grinder-comparison")])], thinking: "One more source before comparing the two.")
        ]
    }

    /// The question the `chat-resolving` story's dock is answering (#3042,
    /// #3130) — sits directly above the dock/reply slot, same as the
    /// working stories' user question, so the reply reads as an answer to
    /// something shown rather than to the collector digest above it.
    static var resolvingUserQuestion: ChatMessage {
        ChatMessage(
            id: 18, serverID: 18, createdAt: Date().addingTimeInterval(-20),
            content: "Can you check the ferry times for this weekend?",
            sourceHint: nil, imageAttachments: [], isOutgoing: true
        )
    }

    /// The reply the `chat-resolving` story's dock resolves into (#3042,
    /// #3130) — carries the working run's own story id so
    /// `beginResolvingIfMatched` matches it, same as a real turn's reply.
    static var resolvingReplyMessage: ChatMessage {
        ChatMessage(
            id: 19, serverID: 19, createdAt: Date(),
            content: "Found three sailings this weekend — cheapest is Friday morning, before the fare bump.",
            sourceHint: nil, runID: "story-run",
            toolCalls: 3, durationMs: 8_000,
            imageAttachmentDataURLs: [tinyLandscapeImageDataURL], imageAttachments: [], isOutgoing: false
        )
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

    /// The bubble story's conversation (#3038), sized to fit one screen:
    /// every Penny bubble carries her image, every bubble has its metadata
    /// line (Penny's with tool count and seconds, the collector's with its
    /// source segment), the user quotes one of Penny's, and the collector
    /// digest carries a table that scrolls horizontally inside its bubble.
    static var bubbleMessages: [ChatMessage] {
        let now = Date()
        func at(_ secondsAgo: TimeInterval) -> Date { now.addingTimeInterval(-secondsAgo) }

        return [
            ChatMessage(
                id: 13, serverID: 13, createdAt: at(140),
                content: "Calm seas — a light south-westerly, so a smooth crossing on the 9:30.",
                sourceHint: nil, runID: "run:bubbles:2",
                durationMs: 3_000,
                imageAttachmentDataURLs: [tinyLandscapeImageDataURL], imageAttachments: [], isOutgoing: false
            ),
            ChatMessage(
                id: 16, serverID: 16, createdAt: at(80),
                content: """
                Can you check the ferry prices for this weekend?

                In reply to Penny:
                > Calm seas — a light south-westerly, so a smooth crossing on the 9:30.
                """,
                sourceHint: nil, parentID: 13, imageAttachments: [], isOutgoing: true
            ),
            ChatMessage(
                id: 17, serverID: 17, createdAt: at(40),
                content: """
                Here's tonight's price-watch digest.

                | Shop | Price | Change |
                | --- | --- | --- |
                | North Coffee | $189 | −$10 |
                | Bean Bros | $205 | — |
                | Harbor Roasters | $212 | +$3 |
                """,
                sourceHint: "Price watch", runID: "run:bubbles:3",
                toolCalls: 3, durationMs: 18_000,
                imageAttachmentDataURLs: [tinyLandscapeImageDataURL], imageAttachments: [], isOutgoing: false
            )
        ]
    }

    /// Find-in-chat's story conversation (#3045): three plain-text
    /// bubbles — no table, no reply-quote — all three containing "ferry",
    /// so the highlight shows cleanly on both bubble colours. Penny's
    /// reply carries her flair image, like every other story's — the app
    /// never sends one of her replies without one.
    static var findInChatMessages: [ChatMessage] {
        let now = Date()
        func at(_ secondsAgo: TimeInterval) -> Date { now.addingTimeInterval(-secondsAgo) }

        return [
            ChatMessage(
                id: 101, serverID: 101, createdAt: at(180),
                content: "What time does the Saturday ferry leave?",
                sourceHint: nil, imageAttachments: [], isOutgoing: true
            ),
            ChatMessage(
                id: 102, serverID: 102, createdAt: at(150),
                content: "The 9:30 ferry has space — want me to book it?",
                sourceHint: nil, imageAttachmentDataURLs: [tinyLandscapeImageDataURL], imageAttachments: [], isOutgoing: false
            ),
            ChatMessage(
                id: 103, serverID: 103, createdAt: at(60),
                content: "Yes, please book the ferry.",
                sourceHint: nil, imageAttachments: [], isOutgoing: true
            )
        ]
    }
}
#endif
