import Foundation
import Observation
import Security
import SwiftUI
import UIKit
import UserNotifications

@MainActor
@Observable
final class PennyWebSocketClient {
    private let urlSession = URLSession(configuration: .default)
    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var heartbeatTask: Task<Void, Never>?
    private var notificationTokenTask: Task<Void, Never>?
    private var localMessageID = -1
    private let databaseService: DatabaseService
    private let prefs: Prefs
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    var messages: [ChatMessage] = []
    var pendingCount = 0
    var isConnected = false
    var isRegistered = false
    var isTyping = false
    var lastError: String?
    
    private let messageLimit = 3
    private let maximumWebSocketMessageSize = 8 * 1024 * 1024

    init() {
        self.databaseService = .shared
        self.prefs = .shared
        loadSavedMessages()
    }

    init(databaseService: DatabaseService) {
        self.databaseService = databaseService
        self.prefs = .shared
        loadSavedMessages()
    }

    init(databaseService: DatabaseService, prefs: Prefs) {
        self.databaseService = databaseService
        self.prefs = prefs
        loadSavedMessages()
    }

    private func loadSavedMessages() {
        databaseService.setup()
        messages = databaseService.loadMessages().map(ChatMessage.init(model:))
        localMessageID = min(-1, (messages.map(\.id).min() ?? 0) - 1)
    }

    var canSend: Bool {
        isConnected && isRegistered
    }

    var statusText: String {
        if let lastError {
            return lastError
        }

        if isRegistered {
            return "Connected"
        }

        if isConnected {
            return "Registering"
        }

        return "Disconnected"
    }

    var connectionColor: Color {
        if isRegistered { return .green }
        if isConnected { return .orange }
        return .red
    }

    func connect() async {
        guard webSocketTask == nil else { return }

        lastError = nil
        guard let request = makeAuthenticatedRequest() else { return }

        let task = urlSession.webSocketTask(with: request)
        task.maximumMessageSize = maximumWebSocketMessageSize
        webSocketTask = task
        task.resume()

        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }
        heartbeatTask = Task { [weak self] in
            await self?.heartbeatLoop()
        }
        notificationTokenTask = Task { [weak self] in
            await self?.notificationTokenLoop()
        }

        sendRegistration()
        send(.pullMessages(limit: messageLimit))
    }

    func reconnect() {
        disconnect()
        Task { await connect() }
    }

    func disconnect() {
        receiveTask?.cancel()
        heartbeatTask?.cancel()
        notificationTokenTask?.cancel()
        receiveTask = nil
        heartbeatTask = nil
        notificationTokenTask = nil
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        isConnected = false
        isRegistered = false
        isTyping = false
    }

    func sendMessage(_ content: String) {
        let message = ChatMessage.local(id: nextLocalMessageID(), content: content)
        messages.append(message)
        databaseService.save(message: MessageModel(message: message))
        send(.message(content: content))
    }

    func makeAuthenticatedRequest() -> URLRequest? {
        guard let path = prefs.webSocketURL, let url = URL(string: path) else {
            lastError = "Invalid WebSocket URL: \(prefs.webSocketURL ?? "none")"
            return nil
        }
        
        guard let username = prefs.username, let password = prefs.password else {
            lastError = "Invalid Username or Password"
            return nil
        }

        var request = URLRequest(url: url)
        let credentials = "\(username):\(password)"
        let encodedCredentials = Data(credentials.utf8).base64EncodedString()
        request.setValue("Basic \(encodedCredentials)", forHTTPHeaderField: "Authorization")
        return request
    }

    private func sendRegistration() {
        send(.register(RegisterPayload.current(apnsToken: PushNotificationState.shared.deviceToken)))
    }

    private func notificationTokenLoop() async {
        let updates = NotificationCenter.default.notifications(named: PushNotificationState.didUpdateDeviceToken)
        for await _ in updates {
            guard !Task.isCancelled else { return }
            sendRegistration()
        }
    }

    private func receiveLoop() async {
        while !Task.isCancelled, let webSocketTask {
            do {
                let incomingMessage = try await webSocketTask.receive()
                try handle(incomingMessage)
            } catch {
                guard !Task.isCancelled else { return }
                lastError = "WebSocket receive failed: \(error.localizedDescription)"
                print(lastError ?? error.localizedDescription)
                isConnected = false
                isRegistered = false
                return
            }
        }
    }

    private func heartbeatLoop() async {
        while !Task.isCancelled {
            do {
                try await Task.sleep(for: .seconds(25))
                send(.heartbeat)
            } catch {
                return
            }
        }
    }

    private func handle(_ incomingMessage: URLSessionWebSocketTask.Message) throws {
        let data: Data
        switch incomingMessage {
        case .data(let messageData):
            data = messageData
        case .string(let messageString):
            print("received \(messageString.utf8)")
            data = Data(messageString.utf8)
        @unknown default:
            return
        }

        let envelope = try decoder.decode(ServerEnvelope.self, from: data)
        switch envelope {
        case .status(let payload):
            isConnected = payload.connected
            lastError = payload.error
        case .registered(let payload):
            isConnected = true
            isRegistered = true
            pendingCount = payload.pendingCount
            send(.pullMessages(limit: messageLimit))
        case .outboxChanged(let payload):
            pendingCount = payload.pendingCount
            if payload.pendingCount > 0 {
                send(.pullMessages(limit: messageLimit))
            }
        case .messages(let payload):
            receive(payload.messages)
        case .messagesAcked:
            break
        case .typing(let payload):
            isTyping = payload.active
        }
    }

    private func receive(_ incomingMessages: [ServerChatMessage]) {
        let incomingIDs = incomingMessages.map(\.id)
        if incomingIDs.isEmpty {
            pendingCount = 0
            return
        }

        let existingIDs = Set(messages.compactMap(\.serverID))
        let newMessages = incomingMessages
            .filter { !existingIDs.contains($0.id) }
            .map(ChatMessage.remote)

        if !newMessages.isEmpty {
            messages.append(contentsOf: newMessages)
            messages.sort { $0.createdAt < $1.createdAt }
            newMessages.forEach { databaseService.save(message: MessageModel(message: $0)) }
        }

        send(.ackMessages(ids: incomingIDs))
        pendingCount = max(0, pendingCount - incomingIDs.count)
        clearAppBadge()

        if pendingCount > 0 {
            send(.pullMessages(limit: messageLimit))
        }
    }

    private func clearAppBadge() {
        UNUserNotificationCenter.current().setBadgeCount(0) { error in
            if let error {
                print("Failed to clear badge count: \(error.localizedDescription)")
            }
        }
    }

    private func send(_ outgoingMessage: ClientMessage) {
        guard let webSocketTask else { return }

        Task {
            do {
                let data = try encoder.encode(outgoingMessage)
                guard let json = String(data: data, encoding: .utf8) else { return }
                print("sent \(json)")
                try await webSocketTask.send(.string(json))
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                }
            }
        }
    }

    private func nextLocalMessageID() -> Int {
        defer { localMessageID -= 1 }
        return localMessageID
    }
}

private enum ClientMessage: Encodable {
    case register(RegisterPayload)
    case message(content: String)
    case pullMessages(limit: Int)
    case ackMessages(ids: [Int])
    case heartbeat

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .register(let payload):
            try container.encode("register", forKey: .type)
            try container.encode(payload.deviceID, forKey: .deviceID)
            try container.encode(payload.label, forKey: .label)
            try container.encodeIfPresent(payload.pairingToken, forKey: .pairingToken)
            try container.encodeIfPresent(payload.deviceSecret, forKey: .deviceSecret)
            try container.encodeIfPresent(payload.apnsToken, forKey: .apnsToken)
            try container.encode(payload.apnsEnvironment, forKey: .apnsEnvironment)
            try container.encode(payload.appVersion, forKey: .appVersion)
        case .message(let content):
            try container.encode("message", forKey: .type)
            try container.encode(content, forKey: .content)
        case .pullMessages(let limit):
            try container.encode("pull_messages", forKey: .type)
            try container.encode(limit, forKey: .limit)
        case .ackMessages(let ids):
            try container.encode("ack_messages", forKey: .type)
            try container.encode(ids, forKey: .ids)
        case .heartbeat:
            try container.encode("heartbeat", forKey: .type)
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case deviceID = "device_id"
        case label
        case pairingToken = "pairing_token"
        case deviceSecret = "device_secret"
        case apnsToken = "apns_token"
        case apnsEnvironment = "apns_environment"
        case appVersion = "app_version"
        case content
        case limit
        case ids
    }
}

private struct RegisterPayload {
    let deviceID: String
    let label: String
    let pairingToken: String?
    let deviceSecret: String?
    let apnsToken: String?
    let apnsEnvironment: String
    let appVersion: String

    static func current(apnsToken: String?) -> RegisterPayload {
        RegisterPayload(
            deviceID: DeviceIdentity.stableDeviceID(),
            label: UIDevice.current.name,
            pairingToken: "pairing-token",
            deviceSecret: DeviceIdentity.deviceSecret(),
            apnsToken: apnsToken,
            apnsEnvironment: ApnsEnvironment.current.rawValue,
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0.0"
        )
    }
}

/// APNs environment this build's push token was minted for.
///
/// A token is only valid against the matching APNs host, so the server must be
/// told which one to use. Historically hardcoded to `sandbox`, which silently
/// broke push on TestFlight/App Store builds (those carry a production token).
///
/// Derivation: DEBUG builds always use the development (sandbox) environment.
/// Release builds read the `aps-environment` entitlement from the embedded
/// provisioning profile — `development` (a dev or ad-hoc signed build, or a
/// direct device install) maps to sandbox; `production` — as in a TestFlight or
/// App Store build, or the absence of an embedded profile — maps to production.
///
/// NOTE: unverified by build on the authoring machine (no Xcode). Verify on a
/// real device / TestFlight build before relying on it.
private enum ApnsEnvironment: String {
    case sandbox
    case production

    static var current: ApnsEnvironment {
        #if DEBUG
        return .sandbox
        #else
        return embeddedProfileEnvironment() ?? .production
        #endif
    }

    private static func embeddedProfileEnvironment() -> ApnsEnvironment? {
        guard
            let url = Bundle.main.url(forResource: "embedded", withExtension: "mobileprovision"),
            let data = try? Data(contentsOf: url),
            let entitlements = provisioningEntitlements(from: data),
            let apsEnvironment = entitlements["aps-environment"] as? String
        else {
            return nil
        }
        return apsEnvironment == "development" ? .sandbox : .production
    }

    private static func provisioningEntitlements(from data: Data) -> [String: Any]? {
        // A .mobileprovision is a CMS (PKCS#7) blob wrapping an XML plist; slice
        // out the plist between the <plist ...> and </plist> markers and parse it.
        guard
            let start = data.range(of: Data("<plist".utf8))?.lowerBound,
            let end = data.range(of: Data("</plist>".utf8))?.upperBound
        else {
            return nil
        }
        let plistData = data.subdata(in: start..<end)
        let profile = try? PropertyListSerialization.propertyList(from: plistData, options: [], format: nil)
        return (profile as? [String: Any])?["Entitlements"] as? [String: Any]
    }
}

private enum ServerEnvelope: Decodable {
    case status(StatusPayload)
    case registered(RegisteredPayload)
    case outboxChanged(OutboxChangedPayload)
    case messages(MessagesPayload)
    case messagesAcked(MessagesAckedPayload)
    case typing(TypingPayload)

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "status":
            self = .status(try StatusPayload(from: decoder))
        case "registered":
            self = .registered(try RegisteredPayload(from: decoder))
        case "outbox_changed":
            self = .outboxChanged(try OutboxChangedPayload(from: decoder))
        case "messages":
            self = .messages(try MessagesPayload(from: decoder))
        case "messages_acked":
            self = .messagesAcked(try MessagesAckedPayload(from: decoder))
        case "typing":
            self = .typing(try TypingPayload(from: decoder))
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "Unknown server message type: \(type)")
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type
    }
}

private struct StatusPayload: Decodable {
    let connected: Bool
    let error: String?
}

private struct RegisteredPayload: Decodable {
    let deviceID: String
    let isDefault: Bool
    let pendingCount: Int

    private enum CodingKeys: String, CodingKey {
        case deviceID = "device_id"
        case isDefault = "is_default"
        case pendingCount = "pending_count"
    }
}

private struct OutboxChangedPayload: Decodable {
    let pendingCount: Int

    private enum CodingKeys: String, CodingKey {
        case pendingCount = "pending_count"
    }
}

private struct MessagesPayload: Decodable {
    let messages: [ServerChatMessage]
}

private struct MessagesAckedPayload: Decodable {
    let count: Int
}

private struct TypingPayload: Decodable {
    let active: Bool
}

private struct ServerChatMessage: Decodable {
    let id: Int
    let createdAt: Date
    let content: String
    let attachments: [Attachment]
    let sourceType: String?
    let sourceName: String?
    let sourceHint: String?
    let pushTitle: String?
    let pushSummary: String?

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

        let createdAtString = try container.decode(String.self, forKey: .createdAt)
        createdAt = DateParser.parse(createdAtString) ?? .now
    }
}

private struct Attachment: Decodable {
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
    let imageAttachmentDataURLs: [String]
    let imageAttachments: [ImageAttachment]
    let isOutgoing: Bool

    var displayTime: String {
        createdAt.formatted(date: .omitted, time: .shortened)
    }

    init(
        id: Int,
        serverID: Int?,
        createdAt: Date,
        content: String,
        sourceHint: String?,
        imageAttachmentDataURLs: [String] = [],
        imageAttachments: [ImageAttachment],
        isOutgoing: Bool
    ) {
        self.id = id
        self.serverID = serverID
        self.createdAt = createdAt
        self.content = content
        self.sourceHint = sourceHint
        self.imageAttachmentDataURLs = imageAttachmentDataURLs
        self.imageAttachments = imageAttachments
        self.isOutgoing = isOutgoing
    }

    init(model: MessageModel) {
        id = model.id
        serverID = model.serverID
        createdAt = model.createdAt
        content = model.content
        sourceHint = model.sourceHint
        imageAttachmentDataURLs = model.imageAttachmentDataURLs
        imageAttachments = model.imageAttachmentDataURLs.compactMap { dataURL in
            guard let data = DataURLDecoder.decode(dataURL), let image = UIImage(data: data) else { return nil }
            return ImageAttachment(image: image)
        }
        isOutgoing = model.isOutgoing
    }

    static func local(id: Int, content: String) -> ChatMessage {
        ChatMessage(id: id, serverID: nil, createdAt: .now, content: content, sourceHint: nil, imageAttachments: [], isOutgoing: true)
    }

    fileprivate static func remote(_ message: ServerChatMessage) -> ChatMessage {
        let imageAttachmentDataURLs = message.attachments.compactMap(\.dataURL)
        let imageAttachments = message.attachments.compactMap(\.image).map(ImageAttachment.init(image:))
        return ChatMessage(id: message.id, serverID: message.id, createdAt: message.createdAt, content: message.content, sourceHint: message.sourceHint, imageAttachmentDataURLs: imageAttachmentDataURLs, imageAttachments: imageAttachments, isOutgoing: false)
    }
}

private enum DataURLDecoder {
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

private enum DateParser {
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

private enum DeviceIdentity {
    private static let service = "PennyClient"
    private static let deviceIDAccount = "device_id"
    private static let deviceSecretAccount = "device_secret"

    static func stableDeviceID() -> String {
        stableUUID(account: deviceIDAccount)
    }

    static func deviceSecret() -> String {
        stableUUID(account: deviceSecretAccount)
    }

    private static func stableUUID(account: String) -> String {
        if let existingValue = readValue(account: account) {
            return existingValue
        }

        let newValue = UUID().uuidString
        saveValue(newValue, account: account)
        return newValue
    }

    private static func readValue(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else { return nil }

        return String(data: data, encoding: .utf8)
    }

    private static func saveValue(_ value: String, account: String) {
        let data = Data(value.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        let attributes: [String: Any] = [kSecValueData as String: data]

        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var addQuery = query
            addQuery[kSecValueData as String] = data
            SecItemAdd(addQuery as CFDictionary, nil)
        }
    }
}
