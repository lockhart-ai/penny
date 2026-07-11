import Foundation

struct MessagePageCursor: Equatable, Sendable {
    let createdAt: Date
    let id: Int
}

extension MessagePageCursor: CustomStringConvertible {
    var description: String {
        "createdAt=\(createdAt.ISO8601Format()), id=\(id)"
    }
}

struct MessagePageRequest: Sendable {
    let limit: Int
    let before: MessagePageCursor?
    let filter: MessagePageFilter

    init(limit: Int = 30, before: MessagePageCursor? = nil, filter: MessagePageFilter = .all) {
        self.limit = limit
        self.before = before
        self.filter = filter
    }
}

struct MessagePage {
    let messages: [ChatMessage]
    let nextCursor: MessagePageCursor?
    let hasMore: Bool
}

struct HistoryPageResult {
    let payload: MessagesPayload
    let savedOrUpdatedCount: Int
}

struct HistorySyncState: Codable, Equatable {
    let channelTypes: [String]
    let includeAttachments: Bool
    var cursor: String?
    var requestedCount: Int
    var savedOrUpdatedCount: Int
    var remainingCount: Int
    var totalCount: Int?
}

enum PennyEmbeddingError: LocalizedError {
    case unavailable(String)
    case invalidResponse
    case disconnected

    var errorDescription: String? {
        switch self {
        case .unavailable(let message):
            return message
        case .invalidResponse:
            return "The server returned an invalid embedding."
        case .disconnected:
            return "Penny is disconnected."
        }
    }
}

enum HistorySyncEvent {
    case page(HistoryPageResult)
    case count(Int)
    case error(String)
}

enum MessagePageFilter: Equatable, Sendable {
    case all
    case penny
    case schedule
    case chat
    case notifier
    case collector

    private static let collectorPrefix = "Collector: "

    var debugDescription: String {
        switch self {
        case .all:
            return "all"
        case .penny:
            return "penny"
        case .schedule:
            return "schedule"
        case .chat:
            return "chat"
        case .notifier:
            return "notifier"
        case .collector:
            return "collector"
        }
    }

    func includes(_ message: ChatMessage) -> Bool {
        switch self {
        case .all:
            return true
        case .penny:
            return ["Penny", "Startup", "Test Push"].contains(message.sourceHint)
        case .schedule:
            return message.sourceHint == "Schedule"
        case .chat:
            return message.isOutgoing || message.sourceHint == "Chat"
        case .notifier:
            return message.sourceHint == "Notifier"
        case .collector:
            return message.sourceHint?.hasPrefix(Self.collectorPrefix) == true
        }
    }
}
