import Foundation
import UIKit

struct ServerChatMessage: Decodable {
    let id: Int
    let createdAt: Date
    let content: String
    let attachments: [Attachment]
    let sourceType: String?
    let sourceName: String?
    let sourceHint: String?
    let pushTitle: String?
    let pushSummary: String?
    let messageID: Int?
    let outboxID: Int?
    let direction: String?
    let channelType: String?
    let deviceLabel: String?
    let deviceIdentifier: String?
    let parentID: Int?
    let embedding: Data?

    var canonicalID: Int { messageID ?? id }

    private enum CodingKeys: String, CodingKey {
        case id
        case createdAt = "created_at"
        case content
        case attachments
        case sourceType = "source_type"
        case sourceName = "source_name"
        case sourceHint = "source_hint"
        case pushTitle = "push_title"
        case pushSummary = "push_summary"
        case messageID = "message_id"
        case outboxID = "outbox_id"
        case direction
        case channelType = "channel_type"
        case deviceLabel = "device_label"
        case deviceIdentifier = "device_identifier"
        case parentID = "parent_id"
        case embedding
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(Int.self, forKey: .id)
        content = try container.decode(String.self, forKey: .content)
        attachments = try container.decodeIfPresent([Attachment].self, forKey: .attachments) ?? []
        sourceType = try container.decodeIfPresent(String.self, forKey: .sourceType)
        sourceName = try container.decodeIfPresent(String.self, forKey: .sourceName)
        sourceHint = try container.decodeIfPresent(String.self, forKey: .sourceHint)
        pushTitle = try container.decodeIfPresent(String.self, forKey: .pushTitle)
        pushSummary = try container.decodeIfPresent(String.self, forKey: .pushSummary)
        messageID = try container.decodeIfPresent(Int.self, forKey: .messageID)
        outboxID = try container.decodeIfPresent(Int.self, forKey: .outboxID)
        direction = try container.decodeIfPresent(String.self, forKey: .direction)
        channelType = try container.decodeIfPresent(String.self, forKey: .channelType)
        deviceLabel = try container.decodeIfPresent(String.self, forKey: .deviceLabel)
        deviceIdentifier = try container.decodeIfPresent(String.self, forKey: .deviceIdentifier)
        parentID = try container.decodeIfPresent(Int.self, forKey: .parentID)
        if let encodedEmbedding = try container.decodeIfPresent(String.self, forKey: .embedding) {
            embedding = Data(base64Encoded: encodedEmbedding)
        } else {
            embedding = nil
        }

        let createdAtString = try container.decode(String.self, forKey: .createdAt)
        createdAt = DateParser.parse(createdAtString) ?? .now
    }
}

struct Attachment: Decodable {
    let dataURL: String?
    let url: String?
    let name: String?
    let contentType: String?

    var image: UIImage? {
        guard let dataURL, let data = DataURLDecoder.decode(dataURL) else { return nil }
        return UIImage(data: data)
    }

    init(from decoder: Decoder) throws {
        if let container = try? decoder.singleValueContainer(), let dataURL = try? container.decode(String.self) {
            self.dataURL = dataURL
            url = nil
            name = nil
            contentType = nil
            return
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        dataURL = try container.decodeIfPresent(String.self, forKey: .dataURL)
        url = try container.decodeIfPresent(String.self, forKey: .url)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        contentType = try container.decodeIfPresent(String.self, forKey: .contentType)
    }

    private enum CodingKeys: String, CodingKey {
        case dataURL = "data_url"
        case url
        case name
        case contentType = "content_type"
    }
}

struct ImageAttachment: Identifiable {
    let id = UUID()
    let image: UIImage
}

struct ChatMessage: Identifiable {
    let id: Int
    let serverID: Int?
    let createdAt: Date
    let content: String
    let sourceHint: String?
    let channelType: String?
    let deviceLabel: String?
    let deviceIdentifier: String?
    let parentID: Int?
    let imageAttachmentDataURLs: [String]
    let imageAttachments: [ImageAttachment]
    let isOutgoing: Bool
    let embedding: Data?

    var displayTime: String {
        createdAt.formatted(date: .omitted, time: .shortened)
    }

    init(
        id: Int,
        serverID: Int?,
        createdAt: Date,
        content: String,
        sourceHint: String?,
        channelType: String? = nil,
        deviceLabel: String? = nil,
        deviceIdentifier: String? = nil,
        parentID: Int? = nil,
        imageAttachmentDataURLs: [String] = [],
        imageAttachments: [ImageAttachment],
        isOutgoing: Bool,
        embedding: Data? = nil
    ) {
        self.id = id
        self.serverID = serverID
        self.createdAt = createdAt
        self.content = content
        self.sourceHint = sourceHint
        self.channelType = channelType
        self.deviceLabel = deviceLabel
        self.deviceIdentifier = deviceIdentifier
        self.parentID = parentID
        self.imageAttachmentDataURLs = imageAttachmentDataURLs
        self.imageAttachments = imageAttachments
        self.isOutgoing = isOutgoing
        self.embedding = embedding
    }

    init(model: MessageModel) {
        id = model.id
        serverID = model.serverID
        createdAt = model.createdAt
        content = model.content
        sourceHint = model.sourceHint
        channelType = model.channelType
        deviceLabel = model.deviceLabel
        deviceIdentifier = model.deviceIdentifier
        parentID = model.parentID
        imageAttachmentDataURLs = model.imageAttachmentDataURLs
        embedding = model.embedding
        imageAttachments = model.imageAttachmentDataURLs.compactMap { dataURL in
            guard let data = DataURLDecoder.decode(dataURL), let image = UIImage(data: data) else { return nil }
            return ImageAttachment(image: image)
        }
        isOutgoing = model.isOutgoing
    }

    static func local(id: Int, content: String) -> ChatMessage {
        ChatMessage(id: id, serverID: nil, createdAt: .now, content: content, sourceHint: nil, imageAttachments: [], isOutgoing: true, embedding: nil)
    }

    static func remote(_ message: ServerChatMessage) -> ChatMessage {
        let imageAttachmentDataURLs = message.attachments.compactMap(\.dataURL)
        let imageAttachments = message.attachments.compactMap(\.image).map(ImageAttachment.init(image:))
        return ChatMessage(
            id: message.canonicalID,
            serverID: message.canonicalID,
            createdAt: message.createdAt,
            content: message.content,
            sourceHint: message.sourceHint,
            channelType: message.channelType,
            deviceLabel: message.deviceLabel,
            deviceIdentifier: message.deviceIdentifier,
            parentID: message.parentID,
            imageAttachmentDataURLs: imageAttachmentDataURLs,
            imageAttachments: imageAttachments,
            isOutgoing: message.direction == "incoming",
            embedding: message.embedding
        )
    }
}

enum DataURLDecoder {
    static func decode(_ value: String) -> Data? {
        let trimmedValue = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let commaIndex = trimmedValue.firstIndex(of: ",") else { return nil }

        let metadata = trimmedValue[..<commaIndex].lowercased()
        guard metadata.hasPrefix("data:"), metadata.contains(";base64") else { return nil }

        let base64StartIndex = trimmedValue.index(after: commaIndex)
        let base64 = trimmedValue[base64StartIndex...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\n", with: "")
            .replacingOccurrences(of: "\r", with: "")

        return Data(base64Encoded: base64)
    }
}

enum DateParser {
    static func parse(_ value: String) -> Date? {
        if let date = iso8601WithFractionalSeconds.date(from: value) {
            return date
        }

        if let date = iso8601.date(from: value) {
            return date
        }

        if let date = localTimestampWithFractionalSeconds.date(from: value) {
            return date
        }

        return localTimestamp.date(from: value)
    }

    private static let iso8601: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private static let iso8601WithFractionalSeconds: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let localTimestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return formatter
    }()

    private static let localTimestampWithFractionalSeconds: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        return formatter
    }()
}
