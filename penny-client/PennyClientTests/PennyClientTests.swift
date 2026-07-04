import Foundation
import Testing
@testable import PennyClient

@Suite(.serialized)
struct PrefsTests {
    @Test func storesTypedValuesInUserDefaults() {
        let userDefaults = makeUserDefaults()
        let prefs = Prefs(userDefaults: userDefaults)

        prefs.set("wss://example.test/penny/", forKey: .webSocketURL)
        prefs.set("robert", forKey: .username)
        prefs.set(true, forKey: "feature.enabled")
        prefs.set(42, forKey: "answer")
        prefs.set(3.14, forKey: "pi")

        #expect(prefs.webSocketURL == "wss://example.test/penny/")
        #expect(prefs.username == "robert")
        #expect(prefs.bool(forKey: "feature.enabled"))
        #expect(prefs.integer(forKey: "answer") == 42)
        #expect(prefs.double(forKey: "pi") == 3.14)
    }

    @Test func storesCodableValuesInUserDefaults() {
        let userDefaults = makeUserDefaults()
        let prefs = Prefs(userDefaults: userDefaults)
        let value = SamplePreference(name: "Penny", count: 3)

        prefs.set(value, forKey: "sample")

        #expect(prefs.value(SamplePreference.self, forKey: "sample") == value)
    }

    @Test func returnsNilConnectionValuesWhenNoDefaultsOrSecretsExist() {
        let userDefaults = makeUserDefaults()
        let prefs = Prefs(userDefaults: userDefaults, bundle: Bundle(for: EmptyBundleMarker.self))

        #expect(prefs.webSocketURL == nil)
        #expect(prefs.username == nil)
        #expect(prefs.password == nil)
    }
}

@Suite(.serialized)
struct DatabaseServiceTests {
    @Test func savesAndLoadsMessagesIncludingAttachments() {
        let database = DatabaseService()
        database.setupForTesting()
        let createdAt = Date(timeIntervalSince1970: 1_783_128_000)
        let model = MessageModel(
            id: 42,
            serverID: 42,
            createdAt: createdAt,
            content: "Hello **Penny**",
            sourceHint: "Chat",
            imageAttachmentDataURLs: ["data:image/png;base64,aGVsbG8="],
            isOutgoing: false
        )

        database.save(message: model)

        let loaded = database.loadMessages()
        #expect(loaded.count == 1)
        #expect(loaded.first?.id == 42)
        #expect(loaded.first?.serverID == 42)
        #expect(loaded.first?.createdAt == createdAt)
        #expect(loaded.first?.content == "Hello **Penny**")
        #expect(loaded.first?.sourceHint == "Chat")
        #expect(loaded.first?.imageAttachmentDataURLs == ["data:image/png;base64,aGVsbG8="])
        #expect(loaded.first?.isOutgoing == false)
    }

    @Test func loadsMessagesInCreationOrder() {
        let database = DatabaseService()
        database.setupForTesting()
        database.save(message: MessageModel(id: 2, serverID: 2, createdAt: Date(timeIntervalSince1970: 20), content: "Second", sourceHint: nil, imageAttachmentDataURLs: [], isOutgoing: false))
        database.save(message: MessageModel(id: 1, serverID: 1, createdAt: Date(timeIntervalSince1970: 10), content: "First", sourceHint: nil, imageAttachmentDataURLs: [], isOutgoing: false))

        let loaded = database.loadMessages()

        #expect(loaded.map(\.content) == ["First", "Second"])
    }
}

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
        let prefs = Prefs(userDefaults: userDefaults, bundle: Bundle(for: EmptyBundleMarker.self))
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
}

@Suite(.serialized)
@MainActor
struct MessageViewModelTests {
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

    @Test func sendDraftIgnoresWhitespaceOnlyDraft() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.draftMessage = "   \n  "

        viewModel.sendDraft()

        #expect(viewModel.draftMessage == "   \n  ")
        #expect(client.messages.isEmpty)
    }

    @Test func keyboardOffsetOnlyAppliesWhenKeyboardVisible() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        #expect(viewModel.keyboardOffset == 0)

        viewModel.keyboardHeight = 300
        #expect(viewModel.keyboardOffset == 276)
    }
}

private struct SamplePreference: Codable, Equatable {
    let name: String
    let count: Int
}

private final class EmptyBundleMarker {}

private func makeUserDefaults() -> UserDefaults {
    let suiteName = "PennyClientTests.\(UUID().uuidString)"
    let userDefaults = UserDefaults(suiteName: suiteName)!
    userDefaults.removePersistentDomain(forName: suiteName)
    return userDefaults
}

@MainActor
private func configuredPrefs(
    url: String? = "wss://example.test/penny/",
    username: String = "alice",
    password: String = "secret"
) -> Prefs {
    let prefs = Prefs(userDefaults: makeUserDefaults(), bundle: Bundle(for: EmptyBundleMarker.self))
    prefs.webSocketURL = url
    prefs.username = username
    prefs.password = password
    return prefs
}

private func configuredDatabase() -> DatabaseService {
    let database = DatabaseService()
    database.setupForTesting()
    return database
}
