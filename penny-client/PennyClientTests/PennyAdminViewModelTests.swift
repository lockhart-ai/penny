import Foundation
import Testing
@testable import PennyClient
@testable import PennyServices

@Suite(.serialized)
@MainActor
struct PennyAdminViewModelTests {
    @Test func appBuildInfoReadsCommitHashOnlyWhenPresent() {
        #expect(AppBuildInfo(infoDictionary: nil).commitHash == nil)
        #expect(AppBuildInfo(infoDictionary: ["PennyBuildCommitHash": "   "]).commitHash == nil)
        #expect(AppBuildInfo(infoDictionary: ["PennyBuildCommitHash": " abc123def456 \n"]).commitHash == "abc123def456")
    }

    @Test func settingsViewModelSavesPrefsAndUsesConfigAndPermissionSurface() async throws {
        let prefs = configuredPrefs(url: "wss://old.example/penny/", username: "alice", password: "secret")
        let (client, transport) = makeAdminClient(prefs: prefs)
        await connectAndClearStartupFrames(client, transport)
        let viewModel = SettingsViewModel(client: client, prefs: prefs)
        #expect(viewModel.canSendTestNotification == false)
        transport.emit("""
        {"type":"registered","device_id":"device","is_default":true,"pending_count":0}
        """)
        #expect(viewModel.canSendTestNotification)
        _ = await sentPayloads(transport, count: 2)
        transport.clearSentPayloads()

        viewModel.refresh()
        transport.emit("""
        {
          "type": "config_response",
          "params": [{
            "key": "LLM_TEMPERATURE",
            "value": "0.7",
            "default": "0.7",
            "description": "Sampling temperature",
            "type": "float",
            "group": "llm"
          }]
        }
        """)
        transport.emit("""
        {
          "type": "domain_permissions_sync",
          "permissions": [{
            "domain": "old.example",
            "permission": "blocked"
          }]
        }
        """)
        transport.emit("""
        {
          "type": "permission_prompt",
          "request_id": "req-1",
          "domain": "new.example",
          "url": "https://new.example/article"
        }
        """)

        let param = try #require(viewModel.runtimeConfigParams.first)
        viewModel.setConfigValue(" 0.2 ", for: param)
        viewModel.saveConfigValue(for: param)
        viewModel.domainDraft = " allow.example "
        viewModel.domainPermission = .allowed
        viewModel.submitDomainPermission()
        let entry = try #require(viewModel.domainPermissions.first)
        viewModel.deleteDomain(entry)
        viewModel.decidePermissionPrompt(allowed: true)
        viewModel.sendTestNotification()

        viewModel.webSocketURL = " wss://new.example/penny/ "
        viewModel.username = " bob "
        viewModel.password = "new-secret"
        viewModel.saveConnection()

        let payloads = await sentPayloads(transport, count: 6)
        #expect(Array(payloads.map(typeName).prefix(6)) == [
            "config_request",
            "config_update",
            "domain_update",
            "domain_delete",
            "permission_decision",
            "test_notification"
        ])
        #expect(payloads[1]["key"] == .string("LLM_TEMPERATURE"))
        #expect(payloads[1]["value"] == .string("0.2"))
        #expect(payloads[2]["domain"] == .string("allow.example"))
        #expect(payloads[2]["permission"] == .string("allowed"))
        #expect(payloads[3]["domain"] == .string("old.example"))
        #expect(payloads[4]["request_id"] == .string("req-1"))
        #expect(payloads[4]["allowed"] == .bool(true))
        #expect(prefs.webSocketURL == "wss://new.example/penny/")
        #expect(prefs.username == "bob")
        #expect(prefs.password == "new-secret")
        client.disconnect()
    }
}

@MainActor
private final class AdminViewModelMockTransport: WebSocketTransport {
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
private func makeAdminClient(
    prefs: Prefs? = nil
) -> (PennyWebSocketClient, AdminViewModelMockTransport) {
    let transport = AdminViewModelMockTransport()
    let client = PennyWebSocketClient(
        databaseService: configuredDatabase(),
        prefs: prefs ?? configuredPrefs(),
        webSocketClient: transport
    )
    return (client, transport)
}

@MainActor
private func connectAndClearStartupFrames(
    _ client: PennyWebSocketClient,
    _ transport: AdminViewModelMockTransport
) async {
    await client.connect()
    _ = await sentPayloads(transport, count: 2)
    transport.clearSentPayloads()
}

@MainActor
private func sentPayloads(
    _ transport: AdminViewModelMockTransport,
    count: Int
) async -> [[String: JSONValue]] {
    for _ in 0..<20 {
        if transport.sentPayloads.count >= count {
            return transport.sentPayloads
        }
        await Task.yield()
    }
    #expect(transport.sentPayloads.count >= count)
    return transport.sentPayloads
}

private func typeName(_ payload: [String: JSONValue]) -> String {
    guard case .string(let type)? = payload["type"] else { return "" }
    return type
}

@Suite(.serialized)
@MainActor
struct ImageAttachmentSettingsViewModelTests {
    @Test func imageTogglesFollowAvailabilityConfirmationAndAutomaticSwitch() async throws {
        let prefs = configuredPrefs()
        let (client, transport) = makeAdminClient(prefs: prefs)
        let viewModel = SettingsViewModel(client: client, prefs: prefs)
        #expect(!viewModel.canEditImageSetting(.automatic))
        await connectAndClearStartupFrames(client, transport)
        transport.emit("""
        {"type":"registered","device_id":"device","is_default":true,"pending_count":0}
        """)
        _ = await sentPayloads(transport, count: 2)
        #expect(viewModel.imageSettingsStatus == "Loading image settings…")
        transport.emit("{\"type\":\"config_response\",\"params\":[]}")
        #expect(!viewModel.canEditImageSetting(.automatic))
        #expect(viewModel.imageSettingsStatus == "Update the server to enable image settings.")
        transport.emit(imageConfigJSON())
        #expect(viewModel.imageSettingsStatus == nil)
        #expect(viewModel.runtimeConfigParams.isEmpty)
        #expect(ImageAttachmentSetting.allCases.allSatisfy { viewModel.canEditImageSetting($0) })
        transport.clearSentPayloads()
        viewModel.setImageSetting(.automatic, enabled: false)
        #expect(!viewModel.imageSettingValue(.automatic))
        #expect(!viewModel.canEditImageSetting(.related))
        #expect(!viewModel.canEditImageSetting(.automatic))
        let payloads = await sentPayloads(transport, count: 1)
        guard case .string(let identifier)? = payloads.first?["request_id"] else {
            Issue.record("Missing image update correlation")
            return
        }
        transport.emit(imageConfigJSON(requestID: identifier, automatic: "0"))
        #expect(viewModel.canEditImageSetting(.automatic))
        #expect(!viewModel.canEditImageSetting(.citedPage))
        #expect(viewModel.imageSettingValue(.citedPage)) // retained while master is off
        viewModel.setImageSetting(.related, enabled: false)
        #expect(client.imageAttachmentSettings.pendingSetting == nil)
        viewModel.setImageSetting(.automatic, enabled: true)
        #expect(viewModel.imageSettingValue(.automatic))
        client.disconnect()
        #expect(!viewModel.canEditImageSetting(.automatic))
        #expect(client.imageAttachmentSettings.error != nil)
    }
}

private func imageConfigJSON(requestID: String? = nil, automatic: String = "1") -> String {
    let params = ImageAttachmentSetting.allCases.map { setting in
        """
        {"key":"\(setting.rawValue)","value":"\(setting == .automatic ? automatic : "1")",
        "default":"1","description":"Image setting","type":"int","group":"iOS Attachments"}
        """
    }.joined(separator: ",")
    let correlation = requestID.map { ",\"request_id\":\"\($0)\"" } ?? ""
    return "{\"type\":\"config_response\",\"params\":[\(params)]\(correlation)}"
}
