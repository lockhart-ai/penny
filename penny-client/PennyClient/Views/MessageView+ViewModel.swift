import Observation
import SwiftUI

extension MessageView {
    enum MessageLayout: Int, CaseIterable, Identifiable, Sendable {
        case message = 1
        case compact = 2
        case media = 3

        var id: Self { self }

        var title: String {
            switch self {
            case .message:
                return "Message"
            case .compact:
                return "Compact"
            case .media:
                return "Media"
            }
        }

        var systemImage: String {
            "\(rawValue).circle"
        }
    }

    enum MessageFilter: String, CaseIterable, Identifiable, Sendable {
        case all
        case penny
        case schedule
        case chat
        case notifier
        case collector

        var id: Self { self }

        var title: String {
            switch self {
            case .all:
                return "All Messages"
            case .penny:
                return "Penny"
            case .schedule:
                return "Schedule"
            case .chat:
                return "Chat"
            case .notifier:
                return "Notifier"
            case .collector:
                return "Collector"
            }
        }

        var systemImage: String {
            switch self {
            case .all:
                return "tray.full"
            case .penny:
                return "sparkles"
            case .schedule:
                return "calendar"
            case .chat:
                return "bubble.left.and.bubble.right"
            case .notifier:
                return "bell"
            case .collector:
                return "tray.and.arrow.down"
            }
        }

        var pageFilter: MessagePageFilter {
            switch self {
            case .all:
                return .all
            case .penny:
                return .penny
            case .schedule:
                return .schedule
            case .chat:
                return .chat
            case .notifier:
                return .notifier
            case .collector:
                return .collector
            }
        }
    }

    @MainActor
    @Observable
    final class ViewModel {
        var client = PennyWebSocketClient()
        var draftMessage = ""
        var isShowingConnectionError = false
        var isShowingSettings = false
        var hasHiddenNewMessages = false
        var selectedMessageLayout: MessageLayout = .message
        var selectedMessageFilter: MessageFilter = .all {
            didSet {
                guard selectedMessageFilter != oldValue else { return }
                if selectedMessageFilter == .all {
                    hasHiddenNewMessages = false
                }
                client.updateLiveMessageFilter(selectedMessageFilter.pageFilter)
                reloadMessagesForSelectedFilter()
            }
        }
        var displayedMessages: [ChatMessage] = [] {
            didSet {
                handleDisplayedMessagesChanged(previousMessages: oldValue)
            }
        }
        var isAtBottom = true
        var isLoadingOlderMessages = false
        var hasMoreOlderMessages = false
        var scrollToBottomRequest = 0

        @ObservationIgnored private let messagePageSize: Int
        @ObservationIgnored private var nextOlderCursor: MessagePageCursor?
        @ObservationIgnored private var pagingTask: Task<Void, Never>?
        @ObservationIgnored private var suppressDisplayedMessageChanges = false
        @ObservationIgnored private var hasBoundLiveMessages = false
        @ObservationIgnored private var olderPagingEnabled = false
        @ObservationIgnored private var messagePagingGeneration = 0
        @ObservationIgnored private var reservedOlderMessageLoad: OlderMessageLoad?
        @ObservationIgnored private var isRestoringOlderMessageScroll = false

        init(client: PennyWebSocketClient? = nil, messagePageSize: Int = 30) {
            let resolvedClient = client ?? PennyWebSocketClient()
            self.client = resolvedClient
            self.messagePageSize = max(1, messagePageSize)
        }

        var shouldShowTypingIndicator: Bool {
            selectedMessageFilter == .all || selectedMessageFilter == .chat
        }

        var canLoadOlderMessages: Bool {
            olderPagingEnabled && hasMoreOlderMessages && !isLoadingOlderMessages && !isRestoringOlderMessageScroll
        }

        func connect() async {
            bindLiveMessagesIfNeeded()
            await loadLatestMessages()
            await client.connect()
        }

        func disconnect() {
            client.unbindLiveMessages()
            hasBoundLiveMessages = false
            client.disconnect()
        }

        func reconnect() {
            bindLiveMessagesIfNeeded()
            client.reconnect()
        }

        func loadLatestMessages() async {
            messagePagingGeneration += 1
            reservedOlderMessageLoad = nil
            isRestoringOlderMessageScroll = false
            isLoadingOlderMessages = false
            olderPagingEnabled = false
            let page = await client.requestMessagePage(MessagePageRequest(limit: messagePageSize, filter: selectedMessageFilter.pageFilter))
            replaceDisplayedMessages(with: page.messages)
            nextOlderCursor = page.nextCursor
            hasMoreOlderMessages = page.hasMore
            scrollToBottomRequest += 1
        }

        func reserveOlderMessageLoad() -> Int? {
            guard canLoadOlderMessages else { return nil }
            guard let nextOlderCursor, let anchorID = displayedMessages.first?.id else { return nil }
            isLoadingOlderMessages = true
            reservedOlderMessageLoad = OlderMessageLoad(
                cursor: nextOlderCursor,
                filter: selectedMessageFilter.pageFilter,
                generation: messagePagingGeneration
            )
            return anchorID
        }

        func loadReservedOlderMessages() async -> Bool {
            guard let reservation = reservedOlderMessageLoad else {
                isLoadingOlderMessages = false
                return false
            }
            defer {
                isLoadingOlderMessages = false
                reservedOlderMessageLoad = nil
            }

            try? await Task.sleep(for: .milliseconds(250))
            guard messagePagingGeneration == reservation.generation,
                  nextOlderCursor == reservation.cursor,
                  selectedMessageFilter.pageFilter == reservation.filter else {
                return false
            }

            let page = await client.requestMessagePage(
                MessagePageRequest(
                    limit: messagePageSize,
                    before: reservation.cursor,
                    filter: reservation.filter
                )
            )
            self.nextOlderCursor = page.nextCursor
            hasMoreOlderMessages = page.hasMore

            let existingIDs = Set(displayedMessages.map(\.id))
            let olderMessages = page.messages.filter { !existingIDs.contains($0.id) }
            guard !olderMessages.isEmpty else { return false }

            suppressDisplayedMessageChanges = true
            displayedMessages.insert(contentsOf: olderMessages, at: 0)
            suppressDisplayedMessageChanges = false
            isRestoringOlderMessageScroll = true
            return true
        }

        func finishOlderMessageScrollRestoration() {
            isRestoringOlderMessageScroll = false
        }

        func enableOlderPaging() {
            olderPagingEnabled = true
        }

        func requestScrollToBottom() {
            olderPagingEnabled = false
            reservedOlderMessageLoad = nil
            isRestoringOlderMessageScroll = false
            isAtBottom = true
            scrollToBottomRequest += 1
        }

        func prepareComposerFocus() {
            if selectedMessageLayout != .message {
                selectedMessageLayout = .message
            }
        }

        func updateBottomVisibility(_ isVisible: Bool) {
            isAtBottom = isVisible
            if isVisible && selectedMessageFilter == .all {
                hasHiddenNewMessages = false
            }
        }

        func waitForPaging() async {
            await pagingTask?.value
        }

        func waitForFiltering() async {
            await waitForPaging()
        }

        func clearFiltersAndShowNewMessages() async {
            hasHiddenNewMessages = false
            if selectedMessageFilter == .all {
                await loadLatestMessages()
            } else {
                selectedMessageFilter = .all
                await waitForPaging()
            }
        }

        func sendDraft() {
            let trimmedMessage = draftMessage.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedMessage.isEmpty else { return }

            draftMessage = ""
            if selectedMessageFilter != .all {
                selectedMessageFilter = .all
            }
            client.sendMessage(trimmedMessage)
            requestScrollToBottom()
        }

        func handleScenePhaseChange(_ phase: ScenePhase) {
            switch phase {
            case .active:
                Task { await connect() }
            case .background:
                disconnect()
            case .inactive:
                break
            @unknown default:
                break
            }
        }

        private func reloadMessagesForSelectedFilter() {
            pagingTask?.cancel()
            pagingTask = Task { [weak self] in
                await self?.loadLatestMessages()
            }
        }

        private func bindLiveMessagesIfNeeded() {
            if hasBoundLiveMessages {
                client.updateLiveMessageFilter(selectedMessageFilter.pageFilter)
                return
            }

            client.bindLiveMessages(
                Binding(
                    get: { [weak self] in self?.displayedMessages ?? [] },
                    set: { [weak self] messages in self?.displayedMessages = messages }
                ),
                hasNewMessages: Binding(
                    get: { [weak self] in self?.hasHiddenNewMessages ?? false },
                    set: { [weak self] hasNewMessages in self?.hasHiddenNewMessages = hasNewMessages }
                ),
                filter: selectedMessageFilter.pageFilter
            )
            hasBoundLiveMessages = true
        }

        private func replaceDisplayedMessages(with messages: [ChatMessage]) {
            suppressDisplayedMessageChanges = true
            displayedMessages = messages
            suppressDisplayedMessageChanges = false
        }

        private func handleDisplayedMessagesChanged(previousMessages: [ChatMessage]) {
            guard !suppressDisplayedMessageChanges else { return }
            guard displayedMessages.count > previousMessages.count else { return }

            if isAtBottom {
                scrollToBottomRequest += 1
            } else {
                hasHiddenNewMessages = true
            }
        }
    }

    private struct OlderMessageLoad {
        let cursor: MessagePageCursor
        let filter: MessagePageFilter
        let generation: Int
    }
}
