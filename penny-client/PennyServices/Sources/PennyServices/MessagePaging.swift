import Foundation

public struct MessagePageCursor: Equatable, Sendable {
    public let createdAt: Date
    public let id: Int
}

extension MessagePageCursor: CustomStringConvertible {
    public var description: String {
        "createdAt=\(createdAt.ISO8601Format()), id=\(id)"
    }
}

public struct MessagePageRequest: Sendable {
    public let limit: Int
    public let before: MessagePageCursor?
    public let filter: MessagePageFilter

    public init(limit: Int = 30, before: MessagePageCursor? = nil, filter: MessagePageFilter = .all) {
        self.limit = limit
        self.before = before
        self.filter = filter
    }
}

public struct MessagePage {
    public let messages: [ChatMessage]
    public let nextCursor: MessagePageCursor?
    public let hasMore: Bool
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

public enum MessagePageFilter: Equatable, Sendable {
    case all
    case penny
    case chat
    case notifier
    case collector

    private static let collectorPrefix = "Collector: "

    public var debugDescription: String {
        switch self {
        case .all:
            return "all"
        case .penny:
            return "penny"
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
        case .chat:
            return message.isOutgoing || message.sourceHint == "Chat"
        case .notifier:
            return message.sourceHint == "Notifier"
        case .collector:
            return message.sourceHint?.hasPrefix(Self.collectorPrefix) == true
        }
    }
}
