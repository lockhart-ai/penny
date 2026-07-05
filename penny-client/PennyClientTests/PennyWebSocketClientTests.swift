import Foundation
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
