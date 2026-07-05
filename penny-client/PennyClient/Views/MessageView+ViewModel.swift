import Observation
import SwiftUI
import UIKit

extension MessageView {
    fileprivate struct MessageFilterInput: Sendable {
        let sourceHint: String?
        let isOutgoing: Bool

        init(message: ChatMessage) {
            sourceHint = message.sourceHint
            isOutgoing = message.isOutgoing
        }
    }

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

        nonisolated private static let collectorPrefix = "Collector: "

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

        nonisolated private var sourceHint: String? {
            switch self {
            case .all, .collector:
                return nil
            case .penny:
                return "Penny"
            case .schedule:
                return "Schedule"
            case .chat:
                return "Chat"
            case .notifier:
                return "Notifier"
            }
        }

        nonisolated fileprivate func includes(_ input: MessageFilterInput) -> Bool {
            switch self {
            case .all:
                return true
            case .penny:
                return ["Penny", "Startup", "Test Push"].contains(input.sourceHint)
            case .chat:
                return input.isOutgoing || input.sourceHint == sourceHint
            case .collector:
                return input.sourceHint?.hasPrefix(Self.collectorPrefix) == true
            default:
                return input.sourceHint == sourceHint
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
        var selectedMessageID: Int?
        var selectedMessageLayout: MessageLayout = .message
        var selectedMessageFilter: MessageFilter = .all {
            didSet {
                if selectedMessageFilter == .all {
                    hasHiddenNewMessages = false
                }
                refreshFilteredMessages()
            }
        }
        var filteredMessages: [ChatMessage]
        var composerHeight: CGFloat = 64
        var keyboardHeight: CGFloat = 0

        @ObservationIgnored private var filterTask: Task<Void, Never>?

        init(client: PennyWebSocketClient? = nil) {
            let resolvedClient = client ?? PennyWebSocketClient()
            self.client = resolvedClient
            filteredMessages = resolvedClient.messages
        }

        private let keyboardComposerSpacing: CGFloat = -24

        var keyboardOffset: CGFloat {
            keyboardHeight > 0 ? keyboardHeight + keyboardComposerSpacing : 0
        }

        var shouldShowTypingIndicator: Bool {
            selectedMessageFilter == .all || selectedMessageFilter == .chat
        }

        func refreshFilteredMessages() {
            let filter = selectedMessageFilter
            let messages = client.messages
            let inputs = messages.map(MessageFilterInput.init)

            filterTask?.cancel()
            filterTask = Task { [weak self] in
                let matchingIndexes = await Task.detached(priority: .userInitiated) {
                    inputs.indices.filter { index in
                        filter.includes(inputs[index])
                    }
                }.value

                guard !Task.isCancelled else { return }
                let filteredMessages = matchingIndexes.map { messages[$0] }
                self?.filteredMessages = filteredMessages
            }
        }

        func waitForFiltering() async {
            await filterTask?.value
        }

        func handleMessagesChanged(previousMessageCount: Int) async -> Bool {
            let messages = client.messages
            let newMessages = messages.dropFirst(min(previousMessageCount, messages.count))
            guard !newMessages.isEmpty else {
                refreshFilteredMessages()
                await waitForFiltering()
                return true
            }

            let filter = selectedMessageFilter
            let inputs = newMessages.map(MessageFilterInput.init)
            let hasVisibleNewMessages = inputs.contains(where: filter.includes)
            let hasFilteredNewMessages = inputs.contains { !filter.includes($0) }

            if hasFilteredNewMessages && filter != .all {
                hasHiddenNewMessages = true
            }

            guard hasVisibleNewMessages else { return false }
            refreshFilteredMessages()
            await waitForFiltering()
            return true
        }

        func clearFiltersAndShowNewMessages() async {
            hasHiddenNewMessages = false
            if selectedMessageFilter == .all {
                refreshFilteredMessages()
            } else {
                selectedMessageFilter = .all
            }
            await waitForFiltering()
        }

        func connect() async {
            await client.connect()
        }

        func disconnect() {
            client.disconnect()
        }

        func reconnect() {
            client.reconnect()
        }

        func sendDraft() {
            let trimmedMessage = draftMessage.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedMessage.isEmpty else { return }

            draftMessage = ""
            client.sendMessage(trimmedMessage)

            if selectedMessageFilter == .all {
                refreshFilteredMessages()
            } else {
                selectedMessageFilter = .all
            }
        }

        func handleScenePhaseChange(_ phase: ScenePhase) {
            switch phase {
            case .active:
                Task { await client.connect() }
            case .background:
                client.disconnect()
            case .inactive:
                break
            @unknown default:
                break
            }
        }

        func updateKeyboardHeight(from notification: Notification) {
            guard let keyboardFrame = notification.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect else { return }

            let screenHeight = UIApplication.shared.connectedScenes
                .compactMap { ($0 as? UIWindowScene)?.screen.bounds.height }
                .first ?? keyboardFrame.maxY
            let overlap = max(0, screenHeight - keyboardFrame.minY)
            let duration = notification.userInfo?[UIResponder.keyboardAnimationDurationUserInfoKey] as? Double ?? 0.25

            withAnimation(.easeOut(duration: duration)) {
                keyboardHeight = overlap
            }
        }
    }
}
