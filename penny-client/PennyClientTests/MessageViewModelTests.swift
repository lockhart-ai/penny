import Foundation
import PennyDesign
import SwiftUI
import Testing
import UIKit
@testable import PennyClient
@testable import PennyServices

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

    @Test func composerFocusReturnsToMessageLayoutWithoutRequestingScroll() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        viewModel.selectedMessageLayout = .media
        let previousScrollRequest = viewModel.scrollToBottomRequest

        let didChangeLayout = viewModel.prepareComposerFocus()

        #expect(didChangeLayout)
        #expect(viewModel.selectedMessageLayout == .message)
        #expect(viewModel.scrollToBottomRequest == previousScrollRequest)
    }

    @Test func composerFocusDoesNotRequestScrollWhenAlreadyUsingMessageLayout() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        let previousScrollRequest = viewModel.scrollToBottomRequest

        let didChangeLayout = viewModel.prepareComposerFocus()

        #expect(didChangeLayout == false)
        #expect(viewModel.selectedMessageLayout == .message)
        #expect(viewModel.scrollToBottomRequest == previousScrollRequest)
    }

    @Test func composerPresentationTracksEveryConversationState() {
        let viewModel = MessageView.ViewModel(
            client: PennyWebSocketClient(
                databaseService: configuredDatabase(),
                prefs: configuredPrefs()
            )
        )

        viewModel.client.conversationState = .elicit
        #expect(viewModel.composerPresentation.placeholder == "Teach Penny")
        #expect(viewModel.composerPresentation.tint == Color.purple.opacity(0.18))

        viewModel.client.conversationState = .learn
        #expect(viewModel.composerPresentation.placeholder == "Approve or correct")
        #expect(viewModel.composerPresentation.tint == Color.blue.opacity(0.16))

        viewModel.client.conversationState = .request
        #expect(viewModel.composerPresentation.placeholder == "Add the missing detail")
        #expect(viewModel.composerPresentation.tint == Color.orange.opacity(0.18))

        for state in [ConversationStateLabel.idle, .apply] {
            viewModel.client.conversationState = state
            #expect(viewModel.composerPresentation.placeholder == "Message")
            #expect(viewModel.composerPresentation.tint == nil)
        }

        viewModel.client.conversationState = nil
        #expect(viewModel.composerPresentation.placeholder == "Message")
        #expect(viewModel.composerPresentation.tint == nil)
    }

    @Test func startupLoadsLatestPageOnly() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...5)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)

        await viewModel.connect()

        #expect(viewModel.displayedMessages.map(\.id) == [4, 5])
        #expect(viewModel.hasMoreOlderMessages)
    }

    @Test func scrollingUpPrependsOlderMessages() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...5)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()

        let anchorID = viewModel.reserveOlderMessageLoad()
        let didLoad = await viewModel.loadReservedOlderMessages()

        #expect(anchorID == 4)
        #expect(didLoad)
        #expect(viewModel.displayedMessages.map(\.id) == [2, 3, 4, 5])
        #expect(viewModel.hasMoreOlderMessages)
    }

    @Test func olderMessageLoadReservationIgnoresDuplicateTriggers() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...5)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()

        let firstAnchorID = viewModel.reserveOlderMessageLoad()
        let secondAnchorID = viewModel.reserveOlderMessageLoad()
        let didLoad = await viewModel.loadReservedOlderMessages()
        let thirdAnchorID = viewModel.reserveOlderMessageLoad()
        viewModel.finishOlderMessageScrollRestoration()
        let fourthAnchorID = viewModel.reserveOlderMessageLoad()

        #expect(firstAnchorID == 4)
        #expect(secondAnchorID == nil)
        #expect(didLoad)
        #expect(thirdAnchorID == nil)
        #expect(fourthAnchorID == 2)
        #expect(viewModel.displayedMessages.map(\.id) == [2, 3, 4, 5])
    }

    @Test func scrollRequestCanSkipLayoutSettling() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        viewModel.requestScrollToBottom(shouldSettleLayout: false)

        #expect(viewModel.shouldSettleScrollToBottom == false)
        #expect(viewModel.scrollToBottomRequest == 1)
    }

    @Test func bottomScrollRequestBlocksOlderPagingUntilScrollLands() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...5)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()

        viewModel.requestScrollToBottom()
        let anchorID = viewModel.reserveOlderMessageLoad()

        #expect(anchorID == nil)
    }

    @Test func filterChangeCancelsReservedOlderMessageLoad() async {
        let database = configuredDatabase()
        saveFilterFixture(in: database)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()

        let anchorID = viewModel.reserveOlderMessageLoad()
        viewModel.selectedMessageFilter = .penny
        await viewModel.waitForPaging()
        let didLoad = await viewModel.loadReservedOlderMessages()

        #expect(anchorID == 7)
        #expect(didLoad == false)
        #expect(viewModel.displayedMessages.map(\.id) == [2, 5])
        #expect(viewModel.hasMoreOlderMessages)
    }

    @Test func filterReloadRequestsBottomScrollAndDisablesOlderPagingUntilScrollLands() async {
        let database = configuredDatabase()
        saveFilterFixture(in: database)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()
        let previousScrollRequest = viewModel.scrollToBottomRequest

        viewModel.selectedMessageFilter = .penny
        await viewModel.waitForPaging()
        let blockedAnchorID = viewModel.reserveOlderMessageLoad()
        viewModel.enableOlderPaging()
        let unblockedAnchorID = viewModel.reserveOlderMessageLoad()

        #expect(viewModel.scrollToBottomRequest == previousScrollRequest + 1)
        #expect(viewModel.hasMoreOlderMessages)
        #expect(blockedAnchorID == nil)
        #expect(unblockedAnchorID == 2)
    }

    @Test func changingFiltersReloadsLatestFilteredPage() async {
        let database = configuredDatabase()
        saveFilterFixture(in: database)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()

        viewModel.selectedMessageFilter = .penny
        await viewModel.waitForPaging()
        #expect(viewModel.displayedMessages.map(\.id) == [1, 2, 5])

        viewModel.selectedMessageFilter = .chat
        await viewModel.waitForPaging()
        #expect(viewModel.displayedMessages.map(\.id) == [4, 8])
        #expect(viewModel.hasMoreOlderMessages == false)
    }

    @Test func sendDraftTrimsMessageClearsDraftAndFilter() async {
        let database = configuredDatabase()
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        viewModel.selectedMessageFilter = .penny
        await viewModel.waitForPaging()
        viewModel.updateBottomVisibility(false)
        let previousScrollRequest = viewModel.scrollToBottomRequest
        viewModel.draftMessage = "  hello Penny  "

        viewModel.sendDraft()
        await viewModel.waitForPaging()

        #expect(viewModel.selectedMessageFilter == .all)
        #expect(viewModel.draftMessage.isEmpty)
        #expect(viewModel.displayedMessages.map(\.content) == ["hello Penny"])
        #expect(database.loadMessages().first?.content == "hello Penny")
        #expect(viewModel.scrollToBottomRequest > previousScrollRequest)
        #expect(viewModel.isAtBottom)
    }

    @Test func sendDraftAppendsReplyContextAndClearsReplyState() async {
        let database = configuredDatabase()
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        let originalMessage = ChatMessage(model: testMessage(
            id: 42,
            content: "First line\n\nSecond line",
            sourceHint: "Chat"
        ))
        viewModel.startReply(to: originalMessage)
        viewModel.draftMessage = "  That makes sense  "

        viewModel.sendDraft()

        #expect(viewModel.replyMessage == nil)
        #expect(viewModel.draftMessage.isEmpty)
        #expect(database.loadMessages().last?.content == """
        That makes sense

        In reply to Chat:
        > First line Second line
        """)
    }

    @Test func cancelReplyClearsReplyState() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        let originalMessage = ChatMessage(model: testMessage(id: 42, content: "Original", sourceHint: "Chat"))

        viewModel.startReply(to: originalMessage)
        viewModel.cancelReply()

        #expect(viewModel.replyMessage == nil)
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

    @Test func sendDraftSendsPendingPhotoWithTheMessageAndClearsIt() async throws {
        let database = configuredDatabase()
        let client = PennyWebSocketClient(databaseService: database, prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.draftMessage = "  What's this bird?  "
        viewModel.pendingImage = solidTestImage()

        viewModel.sendDraft()

        #expect(viewModel.draftMessage.isEmpty)
        #expect(viewModel.pendingImage == nil)
        let sent = try #require(client.messages.last)
        #expect(sent.content == "What's this bird?")
        #expect(sent.imageAttachments.count == 1)
        #expect(sent.imageAttachmentDataURLs.count == 1)
        #expect(sent.imageAttachmentDataURLs.first?.hasPrefix("data:image/jpeg;base64,") == true)
        #expect(database.loadMessages().last?.imageAttachmentDataURLs.count == 1)
    }

    @Test func sendDraftSendsImageOnlyMessage() async throws {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.draftMessage = "   "
        viewModel.pendingImage = solidTestImage()

        viewModel.sendDraft()

        let sent = try #require(client.messages.last)
        #expect(sent.content.isEmpty)
        #expect(sent.imageAttachments.count == 1)
        #expect(viewModel.pendingImage == nil)
    }

    @Test func sendDraftWithEmptyDraftAndNoPhotoSendsNothing() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.draftMessage = "   "

        viewModel.sendDraft()

        #expect(client.messages.isEmpty)
        #expect(viewModel.draftMessage == "   ")
    }

    @Test func visibleLiveMessagesSetBadgeInsteadOfScrollingWhenReadingHistory() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...2)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        viewModel.updateBottomVisibility(false)
        let previousScrollRequest = viewModel.scrollToBottomRequest

        viewModel.client.sendMessage("Live message")

        #expect(viewModel.displayedMessages.map(\.content).last == "Live message")
        #expect(viewModel.hasHiddenNewMessages)
        #expect(viewModel.scrollToBottomRequest == previousScrollRequest)
    }

    @Test func visibleLiveMessagesRequestScrollWhenAlreadyAtBottom() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...2)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        viewModel.updateBottomVisibility(true)
        let previousScrollRequest = viewModel.scrollToBottomRequest

        viewModel.client.sendMessage("Live message")

        #expect(viewModel.displayedMessages.map(\.content).last == "Live message")
        #expect(viewModel.hasHiddenNewMessages == false)
        #expect(viewModel.scrollToBottomRequest == previousScrollRequest + 1)
    }

    @Test func filteredLiveMessagesShowBadgeWithoutAppending() async {
        let database = configuredDatabase()
        let (viewModel, transport) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        viewModel.selectedMessageFilter = .chat
        await viewModel.waitForPaging()

        transport.emit(messageID: 20, content: "Schedule update", sourceHint: "Schedule")

        #expect(viewModel.displayedMessages.isEmpty)
        #expect(viewModel.hasHiddenNewMessages)
        #expect(database.containsMessage(serverID: 20))
    }

    @Test func clearingFiltersShowsHiddenNewMessages() async {
        let database = configuredDatabase()
        let (viewModel, transport) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        viewModel.selectedMessageFilter = .chat
        await viewModel.waitForPaging()
        transport.emit(messageID: 20, content: "Schedule update", sourceHint: "Schedule")

        await viewModel.clearFiltersAndShowNewMessages()

        #expect(viewModel.selectedMessageFilter == .all)
        #expect(viewModel.hasHiddenNewMessages == false)
        #expect(viewModel.displayedMessages.map(\.content) == ["Schedule update"])
    }

    @Test func liveDockVisibilityReflectsSelectedFilter() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        #expect(viewModel.shouldShowLiveDock)

        viewModel.selectedMessageFilter = .chat
        #expect(viewModel.shouldShowLiveDock)

        viewModel.selectedMessageFilter = .penny
        #expect(viewModel.shouldShowLiveDock == false)

        viewModel.selectedMessageFilter = .collector
        #expect(viewModel.shouldShowLiveDock == false)
    }

    // MARK: Find-in-chat (#3045, #3111/#3129 — highlights only, no counter or stepping)

    @Test func searchFindsMatchesNewestFirst() async {
        let database = configuredDatabase()
        saveSearchFixture(in: database)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()

        viewModel.search.open()
        viewModel.search.query = "alpha"

        #expect(viewModel.search.matches.map(\.id) == [5, 3, 1])
        #expect(viewModel.search.currentMatchID == 5)
        #expect(viewModel.search.highlightQuery == "alpha")
    }

    @Test func closingSearchClearsQueryAndMatches() async {
        let database = configuredDatabase()
        saveSearchFixture(in: database)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.search.open()
        viewModel.search.query = "alpha"
        #expect(viewModel.search.matches.isEmpty == false)

        viewModel.search.close()

        #expect(viewModel.search.isSearching == false)
        #expect(viewModel.search.query.isEmpty)
        #expect(viewModel.search.matches.isEmpty)
        #expect(viewModel.search.currentMatchID == nil)
        #expect(viewModel.search.highlightQuery == nil)
    }

    @Test func emptySearchQueryShowsNoMatches() async {
        let database = configuredDatabase()
        saveSearchFixture(in: database)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()

        viewModel.search.open()

        #expect(viewModel.search.matches.isEmpty)
        #expect(viewModel.search.currentMatchID == nil)
        #expect(viewModel.search.highlightQuery == nil)
    }

    @Test func searchPagesOlderHistoryUntilTheSteppedToMatchIsLoaded() async {
        let database = configuredDatabase()
        saveSearchFixture(in: database)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()
        #expect(viewModel.displayedMessages.map(\.id) == [4, 5])
        #expect(viewModel.isSearchMatchLoaded(3) == false)

        let loaded = await viewModel.loadUntilSearchMatchIsPresent(3)

        #expect(loaded)
        #expect(viewModel.isSearchMatchLoaded(3))
        #expect(viewModel.displayedMessages.map(\.id) == [2, 3, 4, 5])
    }

}

@MainActor
private final class MessageViewModelMockTransport: WebSocketTransport {
    private var onReceive: WebSocketTransport.ReceiveHandler?
    private var onFailure: WebSocketTransport.FailureHandler?
    var isConnected = false

    func connect(
        request: URLRequest,
        onReceive: @escaping WebSocketTransport.ReceiveHandler,
        onFailure: @escaping WebSocketTransport.FailureHandler
    ) {
        self.onReceive = onReceive
        self.onFailure = onFailure
        isConnected = true
    }

    func disconnect() {
        onReceive = nil
        onFailure = nil
        isConnected = false
    }

    func send(_ data: Data) async throws {}

    func emit(messageID: Int, content: String, sourceHint: String) {
        onReceive?(Data("""
        {
          "type": "messages",
          "messages": [
            {
              "id": \(messageID),
              "created_at": "2026-07-05T00:00:00Z",
              "content": "\(content)",
              "source_hint": "\(sourceHint)"
            }
          ]
        }
        """.utf8))
    }
}

@MainActor
private func makePagingViewModel(
    database: DatabaseService,
    pageSize: Int
) -> (MessageView.ViewModel, MessageViewModelMockTransport) {
    let transport = MessageViewModelMockTransport()
    let client = PennyWebSocketClient(
        databaseService: database,
        prefs: configuredPrefs(),
        webSocketClient: transport
    )
    return (MessageView.ViewModel(client: client, messagePageSize: pageSize), transport)
}

private func saveFilterFixture(in database: DatabaseService) {
    database.save(message: testMessage(id: 1, content: "Penny", sourceHint: "Penny"))
    database.save(message: testMessage(id: 2, content: "Startup", sourceHint: "Startup"))
    database.save(message: testMessage(id: 3, content: "Schedule", sourceHint: "Schedule"))
    database.save(message: testMessage(id: 4, content: "Chat", sourceHint: "Chat"))
    database.save(message: testMessage(id: 5, content: "Test Push", sourceHint: "Test Push"))
    database.save(message: testMessage(id: 6, content: "Notifier", sourceHint: "Notifier"))
    database.save(message: testMessage(id: 7, content: "Collector", sourceHint: "Collector: flight-deals"))
    database.save(message: testMessage(id: 8, serverID: nil, content: "Outgoing", isOutgoing: true))
}

private func saveNumberedMessages(in database: DatabaseService, ids: ClosedRange<Int>) {
    for id in ids {
        database.save(message: testMessage(id: id, content: "Message \(id)"))
    }
}

/// Five messages, newest last (ids are seconds-since-epoch, #testMessage),
/// three of them containing "alpha" — find-in-chat's view-model tests
/// (#3045).
private func saveSearchFixture(in database: DatabaseService) {
    database.save(message: testMessage(id: 1, content: "alpha one"))
    database.save(message: testMessage(id: 2, content: "bravo"))
    database.save(message: testMessage(id: 3, content: "alpha two"))
    database.save(message: testMessage(id: 4, content: "charlie"))
    database.save(message: testMessage(id: 5, content: "alpha three"))
}

/// A solid-colour photo for the composer's pending-image tests (#3039).
@MainActor
private func solidTestImage() -> UIImage {
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let renderer = UIGraphicsImageRenderer(size: CGSize(width: 120, height: 80), format: format)
    return renderer.image { context in
        UIColor.systemTeal.setFill()
        context.fill(CGRect(x: 0, y: 0, width: 120, height: 80))
    }
}

private func testMessage(
    id: Int,
    serverID: Int? = nil,
    content: String,
    sourceHint: String? = nil,
    isOutgoing: Bool = false
) -> MessageModel {
    MessageModel(
        id: id,
        serverID: serverID ?? (isOutgoing ? nil : id),
        createdAt: Date(timeIntervalSince1970: TimeInterval(id)),
        content: content,
        sourceHint: sourceHint,
        imageAttachmentDataURLs: [],
        isOutgoing: isOutgoing
    )
}
