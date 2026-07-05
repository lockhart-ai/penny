import Foundation
import Testing
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

    @Test func composerFocusReturnsToMessageLayoutWithoutRequestingScroll() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        viewModel.selectedMessageLayout = .media
        let previousScrollRequest = viewModel.scrollToBottomRequest

        viewModel.prepareComposerFocus()

        #expect(viewModel.selectedMessageLayout == .message)
        #expect(viewModel.scrollToBottomRequest == previousScrollRequest)
    }

    @Test func composerFocusDoesNotRequestScrollWhenAlreadyUsingMessageLayout() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        let previousScrollRequest = viewModel.scrollToBottomRequest

        viewModel.prepareComposerFocus()

        #expect(viewModel.selectedMessageLayout == .message)
        #expect(viewModel.scrollToBottomRequest == previousScrollRequest)
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
