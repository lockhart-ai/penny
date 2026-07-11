import Foundation
import SwiftUI
import Testing
@testable import PennyClient

@Suite(.serialized)
@MainActor
struct PennyWebSocketClientTests {
    @Test func buildsAuthenticatedRequestFromPrefs() {
        let prefs = configuredPrefs(url: "wss://example.test/penny/", username: "alice", password: "secret")
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: prefs)

        let request = client.makeAuthenticatedRequest()

        #expect(request?.url?.absoluteString == "wss://example.test/penny/")
        #expect(request?.value(forHTTPHeaderField: "Authorization") == "Basic YWxpY2U6c2VjcmV0")
    }

    @Test func reportsInvalidWebSocketURL() {
        let prefs = configuredPrefs(url: nil, username: "alice", password: "secret")
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: prefs)

        let request = client.makeAuthenticatedRequest()

        #expect(request == nil)
        #expect(client.lastError == "Invalid WebSocket URL: none")
    }

    @Test func reportsMissingCredentials() {
        let userDefaults = makeUserDefaults()
        let prefs = Prefs(userDefaults: userDefaults, keychain: InMemoryKeychain(), bundle: Bundle(for: EmptyBundleMarker.self))
        prefs.webSocketURL = "wss://example.test/penny/"
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: prefs)

        let request = client.makeAuthenticatedRequest()

        #expect(request == nil)
        #expect(client.lastError == "Invalid Username or Password")
    }

    @Test func sendMessageAppendsAndPersistsLocalMessage() {
        let database = configuredDatabase()
        let client = PennyWebSocketClient(databaseService: database, prefs: configuredPrefs())

        client.sendMessage("hello Penny")

        #expect(client.messages.count == 1)
        #expect(client.messages.first?.content == "hello Penny")
        #expect(client.messages.first?.isOutgoing == true)
        #expect(database.loadMessages().first?.content == "hello Penny")
    }

    @Test func initializationDoesNotLoadSavedMessagesIntoMemory() async {
        let database = configuredDatabase()
        database.save(message: MessageModel(id: 1, serverID: 1, createdAt: Date(timeIntervalSince1970: 1), content: "Saved", sourceHint: "Chat", imageAttachmentDataURLs: [], isOutgoing: false))
        let client = PennyWebSocketClient(databaseService: database, prefs: configuredPrefs())

        let page = await client.requestMessagePage(MessagePageRequest(limit: 20, filter: .all))

        #expect(client.messages.isEmpty)
        #expect(page.messages.map(\.content) == ["Saved"])
    }

    @Test func sendMessagePublishesToBoundMessages() {
        let database = configuredDatabase()
        let client = PennyWebSocketClient(databaseService: database, prefs: configuredPrefs())
        var liveMessages: [ChatMessage] = []
        var hasNewMessages = false
        client.bindLiveMessages(
            Binding(get: { liveMessages }, set: { liveMessages = $0 }),
            hasNewMessages: Binding(get: { hasNewMessages }, set: { hasNewMessages = $0 }),
            filter: .all
        )

        client.sendMessage("hello Penny")

        #expect(liveMessages.map(\.content) == ["hello Penny"])
        #expect(hasNewMessages == false)
        #expect(database.loadMessages().first?.content == "hello Penny")
    }

    @Test func embeddingRequestUsesCorrelationIDAndDecodesResponse() async throws {
        let transport = MessagePagingMockTransport()
        let client = PennyWebSocketClient(
            databaseService: configuredDatabase(),
            prefs: configuredPrefs(),
            webSocketClient: transport
        )

        await client.connect()
        _ = await sentPayloads(transport, count: 2)
        transport.emit("""
        {"type":"registered","device_id":"device","is_default":true,"pending_count":0}
        """)

        let requestTask = Task { try await client.requestEmbedding("coffee") }
        let payloads = await sentPayloads(transport, count: 3)
        guard case .string(let requestID)? = payloads.last?["request_id"] else {
            Issue.record("Embedding request did not include a request_id")
            return
        }
        #expect(payloads.last?["type"] == .string("embedding_request"))
        #expect(payloads.last?["text"] == .string("coffee"))

        transport.emit("""
        {"type":"embedding_response","request_id":"\(requestID)","embedding":"AACAPwAAAD8AAIC/"}
        """)

        #expect(try await requestTask.value == Data([0, 0, 128, 63, 0, 0, 0, 63, 0, 0, 128, 191]))
    }

    @Test func receivedMessagesPersistAckDedupeAndPublishLiveMessages() async {
        let database = configuredDatabase()
        let transport = MessagePagingMockTransport()
        let client = PennyWebSocketClient(databaseService: database, prefs: configuredPrefs(), webSocketClient: transport)
        var liveMessages: [ChatMessage] = []
        var hasNewMessages = false
        client.bindLiveMessages(
            Binding(get: { liveMessages }, set: { liveMessages = $0 }),
            hasNewMessages: Binding(get: { hasNewMessages }, set: { hasNewMessages = $0 }),
            filter: .chat
        )
        await client.connect()
        _ = await sentPayloads(transport, count: 2)
        transport.clearSentPayloads()

        transport.emit("""
        {
          "type": "messages",
          "messages": [
            {
              "id": 11,
              "created_at": "2026-07-05T00:00:00Z",
              "content": "Visible chat",
              "source_hint": "Chat"
            },
            {
              "id": 12,
              "created_at": "2026-07-05T00:00:01Z",
              "content": "Filtered schedule",
              "source_hint": "Schedule"
            }
          ]
        }
        """)
        let payloads = await sentPayloads(transport, count: 1)

        #expect(liveMessages.map(\.content) == ["Visible chat"])
        #expect(hasNewMessages)
        #expect(database.containsMessage(serverID: 11))
        #expect(database.containsMessage(serverID: 12))
        #expect(payloads.last?["type"] == .string("ack_messages"))
        #expect(payloads.last?["ids"] == .array([.number(11), .number(12)]))

        transport.clearSentPayloads()
        transport.emit("""
        {
          "type": "messages",
          "messages": [
            {
              "id": 11,
              "created_at": "2026-07-05T00:00:00Z",
              "content": "Visible chat duplicate",
              "source_hint": "Chat"
            }
          ]
        }
        """)
        _ = await sentPayloads(transport, count: 1)

        #expect(liveMessages.map(\.content) == ["Visible chat"])
        #expect(database.loadMessages().filter { $0.serverID == 11 }.count == 1)
    }

    @Test func duplicateMessagesInOnePayloadBecomeOnePersistedAndDisplayedMessage() async {
        let database = configuredDatabase()
        let transport = MessagePagingMockTransport()
        let client = PennyWebSocketClient(
            databaseService: database,
            prefs: configuredPrefs(),
            webSocketClient: transport
        )
        var liveMessages: [ChatMessage] = []
        var hasNewMessages = false
        client.bindLiveMessages(
            Binding(get: { liveMessages }, set: { liveMessages = $0 }),
            hasNewMessages: Binding(get: { hasNewMessages }, set: { hasNewMessages = $0 }),
            filter: .all
        )

        await client.connect()
        transport.clearSentPayloads()
        transport.emit("""
        {
          "type": "messages",
          "messages": [
            {
              "id": 41,
              "created_at": "2026-07-05T00:00:01Z",
              "content": "Penny response",
              "source_hint": "Penny"
            },
            {
              "id": 41,
              "created_at": "2026-07-05T00:00:01Z",
              "content": "Penny response duplicate",
              "source_hint": "Penny"
            }
          ]
        }
        """)

        #expect(liveMessages.map(\.id) == [41])
        #expect(database.loadMessages().filter { $0.serverID == 41 }.count == 1)
        #expect(client.messages.map(\.id) == [41])
        #expect(hasNewMessages == false)
    }

    @Test func historyReconcilesLocallyStoredOutgoingMessage() async {
        let database = configuredDatabase()
        let transport = MessagePagingMockTransport()
        let client = PennyWebSocketClient(
            databaseService: database,
            prefs: configuredPrefs(),
            webSocketClient: transport
        )
        var liveMessages: [ChatMessage] = []
        var hasNewMessages = false
        client.bindLiveMessages(
            Binding(get: { liveMessages }, set: { liveMessages = $0 }),
            hasNewMessages: Binding(get: { hasNewMessages }, set: { hasNewMessages = $0 }),
            filter: .all
        )
        await client.connect()
        client.sendMessage("hi")
        #expect(liveMessages.map(\.id) == [-1])

        transport.emit("""
        {
          "type": "messages",
          "mode": "history",
          "messages": [
            {
              "id": 51,
              "created_at": "2026-07-05T00:00:01Z",
              "content": "hi",
              "direction": "incoming",
              "source_hint": "Chat"
            }
          ]
        }
        """)

        #expect(liveMessages.map(\.id) == [51])
        #expect(client.messages.map(\.id) == [51])
        #expect(database.loadMessages().map(\.id) == [51])
        #expect(hasNewMessages == false)
    }

    @Test func outboxDeliveryPublishesMessageAlreadyPersistedByHistory() async {
        let database = configuredDatabase()
        database.save(message: MessageModel(
            id: 21,
            serverID: 21,
            createdAt: Date(timeIntervalSince1970: 21),
            content: "Historical message",
            sourceHint: "Penny",
            imageAttachmentDataURLs: [],
            isOutgoing: false
        ))
        let transport = MessagePagingMockTransport()
        let client = PennyWebSocketClient(
            databaseService: database,
            prefs: configuredPrefs(),
            webSocketClient: transport
        )
        var liveMessages: [ChatMessage] = []
        var hasNewMessages = false
        client.bindLiveMessages(
            Binding(get: { liveMessages }, set: { liveMessages = $0 }),
            hasNewMessages: Binding(get: { hasNewMessages }, set: { hasNewMessages = $0 }),
            filter: .all
        )

        await client.connect()
        _ = await sentPayloads(transport, count: 2)
        transport.emit("""
        {
          "type": "messages",
          "messages": [
            {
              "id": 7,
              "message_id": 21,
              "outbox_id": 7,
              "created_at": "2026-07-05T00:00:00Z",
              "content": "Historical message",
              "source_hint": "Penny"
            }
          ]
        }
        """)

        _ = await sentPayloads(transport, count: 1)
        // Keep the assertion focused on the live binding rather than persistence.
        #expect(liveMessages.map(\.content) == ["Historical message"])
        #expect(hasNewMessages == false)
    }

    @Test func historyMarksUserSentMessagesAsOutgoing() async {
        let database = configuredDatabase()
        let transport = MessagePagingMockTransport()
        let client = PennyWebSocketClient(
            databaseService: database,
            prefs: configuredPrefs(),
            webSocketClient: transport
        )
        var liveMessages: [ChatMessage] = []
        var hasNewMessages = false
        client.bindLiveMessages(
            Binding(get: { liveMessages }, set: { liveMessages = $0 }),
            hasNewMessages: Binding(get: { hasNewMessages }, set: { hasNewMessages = $0 }),
            filter: .all
        )
        await client.connect()
        _ = await sentPayloads(transport, count: 2)
        transport.clearSentPayloads()
        client.isConnected = true
        client.isRegistered = true
        client.startHistorySync(channelTypes: ["ios"])
        _ = await sentPayloads(transport, count: 1)

        transport.emit("""
        {
          "type": "messages",
          "mode": "history_count",
          "total_count": 1,
          "messages": []
        }
        """)
        _ = await sentPayloads(transport, count: 2)

        transport.emit("""
        {
          "type": "messages",
          "mode": "history",
          "has_more": false,
          "messages": [
            {
              "id": 31,
              "message_id": 31,
              "created_at": "2026-07-05T00:00:00Z",
              "content": "Message from me",
              "direction": "incoming",
              "source_hint": "Chat"
            }
          ]
        }
        """)

        try? await Task.sleep(for: .milliseconds(20))

        #expect(liveMessages.first?.isOutgoing == true)
        #expect(database.loadMessages().first?.isOutgoing == true)
        #expect(hasNewMessages == false)
    }

    @Test func connectionStatusReflectsState() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())

        #expect(client.statusText == "Disconnected")
        #expect(client.canSend == false)

        client.isConnected = true
        #expect(client.statusText == "Registering")

        client.isRegistered = true
        #expect(client.statusText == "Connected")
        #expect(client.canSend)

        client.lastError = "boom"
        #expect(client.statusText == "boom")
    }

    @Test func disconnectClearsConnectionState() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        client.isConnected = true
        client.isRegistered = true
        client.isTyping = true

        client.disconnect()

        #expect(client.isConnected == false)
        #expect(client.isRegistered == false)
        #expect(client.isTyping == false)
        #expect(client.canSend == false)
    }
}

@MainActor
private final class MessagePagingMockTransport: WebSocketTransport {
    private(set) var sentPayloads: [[String: JSONValue]] = []
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

    func send(_ data: Data) async throws {
        sentPayloads.append(try JSONDecoder().decode([String: JSONValue].self, from: data))
    }

    func emit(_ json: String) {
        onReceive?(Data(json.utf8))
    }

    func clearSentPayloads() {
        sentPayloads.removeAll()
    }
}

@MainActor
private func sentPayloads(
    _ transport: MessagePagingMockTransport,
    count: Int
) async -> [[String: JSONValue]] {
    for _ in 0..<100 {
        if transport.sentPayloads.count >= count {
            return transport.sentPayloads
        }
        try? await Task.sleep(for: .milliseconds(10))
    }
    #expect(transport.sentPayloads.count >= count)
    return transport.sentPayloads
}
