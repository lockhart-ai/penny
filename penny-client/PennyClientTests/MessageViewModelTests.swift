import Foundation
import Testing
import UIKit
@testable import PennyClient

@Suite(.serialized)
@MainActor
struct MessageViewModelTests {
    @Test func messageLayoutDefaultsToCurrentMessageStyle() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        #expect(viewModel.selectedMessageLayout == .message)
    }

    @Test func messageLayoutCanSwitchBetweenAvailableLayouts() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        viewModel.selectedMessageLayout = .compact
        #expect(viewModel.selectedMessageLayout == .compact)

        viewModel.selectedMessageLayout = .media
        #expect(viewModel.selectedMessageLayout == .media)
    }

    @Test func sendDraftTrimsMessageAndClearsDraft() {
        let database = configuredDatabase()
        let client = PennyWebSocketClient(databaseService: database, prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.draftMessage = "  hello Penny  "

        viewModel.sendDraft()

        #expect(viewModel.draftMessage.isEmpty)
        #expect(client.messages.first?.content == "hello Penny")
        #expect(database.loadMessages().first?.content == "hello Penny")
    }

    @Test func sendDraftClearsFilters() async {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.selectedMessageFilter = .penny
        viewModel.draftMessage = "hello Penny"

        viewModel.sendDraft()
        await viewModel.waitForFiltering()

        #expect(viewModel.selectedMessageFilter == .all)
        #expect(viewModel.filteredMessages.map(\.content) == ["hello Penny"])
    }

    @Test func sendDraftIgnoresWhitespaceOnlyDraft() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.selectedMessageFilter = .penny
        viewModel.draftMessage = "   \n  "

        viewModel.sendDraft()

        #expect(viewModel.selectedMessageFilter == .penny)
        #expect(viewModel.draftMessage == "   \n  ")
        #expect(client.messages.isEmpty)
    }

    @Test func filteredMessagesReflectSelectedFilter() async {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        client.messages = [
            ChatMessage(id: 1, serverID: 1, createdAt: Date(timeIntervalSince1970: 1), content: "Penny", sourceHint: "Penny", imageAttachments: [], isOutgoing: false),
            ChatMessage(id: 2, serverID: 2, createdAt: Date(timeIntervalSince1970: 2), content: "Startup", sourceHint: "Startup", imageAttachments: [], isOutgoing: false),
            ChatMessage(id: 3, serverID: 3, createdAt: Date(timeIntervalSince1970: 3), content: "Schedule", sourceHint: "Schedule", imageAttachments: [], isOutgoing: false),
            ChatMessage(id: 4, serverID: 4, createdAt: Date(timeIntervalSince1970: 4), content: "Chat", sourceHint: "Chat", imageAttachments: [], isOutgoing: false),
            ChatMessage(id: 5, serverID: 5, createdAt: Date(timeIntervalSince1970: 5), content: "Test Push", sourceHint: "Test Push", imageAttachments: [], isOutgoing: false),
            ChatMessage(id: 6, serverID: 6, createdAt: Date(timeIntervalSince1970: 6), content: "Notifier", sourceHint: "Notifier", imageAttachments: [], isOutgoing: false),
            ChatMessage(id: 7, serverID: 7, createdAt: Date(timeIntervalSince1970: 7), content: "Collector", sourceHint: "Collector: flight-deals", imageAttachments: [], isOutgoing: false),
            ChatMessage(id: 8, serverID: nil, createdAt: Date(timeIntervalSince1970: 8), content: "Outgoing", sourceHint: nil, imageAttachments: [], isOutgoing: true)
        ]
        let viewModel = MessageView.ViewModel(client: client)

        #expect(viewModel.filteredMessages.map(\.id) == [1, 2, 3, 4, 5, 6, 7, 8])

        viewModel.selectedMessageFilter = .penny
        await viewModel.waitForFiltering()
        #expect(viewModel.filteredMessages.map(\.id) == [1, 2, 5])

        viewModel.selectedMessageFilter = .schedule
        await viewModel.waitForFiltering()
        #expect(viewModel.filteredMessages.map(\.id) == [3])

        viewModel.selectedMessageFilter = .chat
        await viewModel.waitForFiltering()
        #expect(viewModel.filteredMessages.map(\.id) == [4, 8])

        viewModel.selectedMessageFilter = .notifier
        await viewModel.waitForFiltering()
        #expect(viewModel.filteredMessages.map(\.id) == [6])

        viewModel.selectedMessageFilter = .collector
        await viewModel.waitForFiltering()
        #expect(viewModel.filteredMessages.map(\.id) == [7])
    }

    @Test func hiddenNewMessagesDoNotRequestScroll() async {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        client.messages = [
            ChatMessage(id: 1, serverID: 1, createdAt: Date(timeIntervalSince1970: 1), content: "Chat", sourceHint: "Chat", imageAttachments: [], isOutgoing: false)
        ]
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.selectedMessageFilter = .chat
        await viewModel.waitForFiltering()
        client.messages.append(ChatMessage(id: 2, serverID: 2, createdAt: Date(timeIntervalSince1970: 2), content: "Schedule", sourceHint: "Schedule", imageAttachments: [], isOutgoing: false))

        let shouldScroll = await viewModel.handleMessagesChanged(previousMessageCount: 1)

        #expect(shouldScroll == false)
        #expect(viewModel.hasHiddenNewMessages)
        #expect(viewModel.filteredMessages.map(\.id) == [1])
    }

    @Test func visibleNewMessagesRequestScrollAfterFiltering() async {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        client.messages = [
            ChatMessage(id: 1, serverID: 1, createdAt: Date(timeIntervalSince1970: 1), content: "Chat", sourceHint: "Chat", imageAttachments: [], isOutgoing: false)
        ]
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.selectedMessageFilter = .chat
        await viewModel.waitForFiltering()
        client.messages.append(ChatMessage(id: 2, serverID: nil, createdAt: Date(timeIntervalSince1970: 2), content: "Outgoing", sourceHint: nil, imageAttachments: [], isOutgoing: true))

        let shouldScroll = await viewModel.handleMessagesChanged(previousMessageCount: 1)

        #expect(shouldScroll)
        #expect(viewModel.hasHiddenNewMessages == false)
        #expect(viewModel.filteredMessages.map(\.id) == [1, 2])
    }

    @Test func clearingFiltersShowsHiddenNewMessages() async {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        client.messages = [
            ChatMessage(id: 1, serverID: 1, createdAt: Date(timeIntervalSince1970: 1), content: "Chat", sourceHint: "Chat", imageAttachments: [], isOutgoing: false)
        ]
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.selectedMessageFilter = .chat
        await viewModel.waitForFiltering()
        client.messages.append(ChatMessage(id: 2, serverID: 2, createdAt: Date(timeIntervalSince1970: 2), content: "Schedule", sourceHint: "Schedule", imageAttachments: [], isOutgoing: false))
        _ = await viewModel.handleMessagesChanged(previousMessageCount: 1)

        await viewModel.clearFiltersAndShowNewMessages()

        #expect(viewModel.selectedMessageFilter == .all)
        #expect(viewModel.hasHiddenNewMessages == false)
        #expect(viewModel.filteredMessages.map(\.id) == [1, 2])
    }

    @Test func typingIndicatorVisibilityReflectsSelectedFilter() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        #expect(viewModel.shouldShowTypingIndicator)

        viewModel.selectedMessageFilter = .chat
        #expect(viewModel.shouldShowTypingIndicator)

        viewModel.selectedMessageFilter = .penny
        #expect(viewModel.shouldShowTypingIndicator == false)

        viewModel.selectedMessageFilter = .collector
        #expect(viewModel.shouldShowTypingIndicator == false)
    }

    @Test func keyboardOffsetOnlyAppliesWhenKeyboardVisible() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        #expect(viewModel.keyboardOffset == 0)

        viewModel.keyboardHeight = 300
        #expect(viewModel.keyboardOffset == 276)
    }
}
