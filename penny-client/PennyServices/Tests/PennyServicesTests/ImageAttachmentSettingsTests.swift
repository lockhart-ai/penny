import Foundation
import Testing
@testable import PennyServices

@Suite(.serialized)
@MainActor
struct ImageAttachmentSettingsTests {
    @Test func confirmsOnlyMatchingUpdatesAndRefreshesOnReconnect() async throws {
        let (service, transport) = await subject()
        #expect(transport.payloads.contains { $0["type"] == .string("config_request") })
        #expect(!service.supportsImageAttachmentSettings)
        transport.emit(configJSON())
        #expect(service.supportsImageAttachmentSettings)
        transport.payloads = []
        service.updateImageAttachmentSetting(.automatic, enabled: false)
        let identifier = try #require(service.imageAttachmentSettings.requestID)
        await drainTasks()
        #expect(transport.payloads == [[
            "type": .string("config_update"), "key": .string("IOS_AUTOMATIC_IMAGES"),
            "value": .string("0"), "request_id": .string(identifier)
        ]])
        transport.emit(configJSON()) // an unrelated refresh is not an acknowledgement
        #expect(service.imageAttachmentSettings.pendingSetting == .automatic)
        service.updateImageAttachmentSetting(.related, enabled: false)
        #expect(service.imageAttachmentSettings.requestID == identifier)
        transport.emit(configJSON(requestID: identifier, automatic: "0"))
        #expect(service.imageAttachmentSettings.pendingSetting == nil)
        #expect(service.imageAttachmentSettings.error == nil)
        #expect(service.runtimeConfigParams.first?.value == "0")
        service.disconnect()
        #expect(!service.supportsImageAttachmentSettings)
        #expect(service.runtimeConfigParams.isEmpty)
        await service.connect()
        transport.emit(registeredJSON)
        await drainTasks()
        #expect(!service.hasLoadedRuntimeConfig)
        transport.emit(configJSON(automatic: "0"))
        #expect(service.runtimeConfigParams.first?.value == "0")
        service.disconnect()
    }

    @Test func rejectionTimeoutAndLateResponsesPreserveConfirmedValues() async throws {
        let (service, transport) = await subject()
        transport.emit(configJSON())
        service.updateImageAttachmentSetting(.automatic, enabled: false)
        let rejectedID = try #require(service.imageAttachmentSettings.requestID)
        transport.emit(configJSON(requestID: rejectedID, error: "Setting rejected"))
        #expect(service.imageAttachmentSettings.error == "Setting rejected")
        #expect(service.runtimeConfigParams.first?.value == "1")
        service.updateImageAttachmentSetting(.automatic, enabled: false)
        let expiredID = try #require(service.imageAttachmentSettings.requestID)
        service.imageAttachmentSettings.fail(requestID: expiredID, message: "Timed out")
        #expect(service.imageAttachmentSettings.error == "Timed out")
        service.updateImageAttachmentSetting(.automatic, enabled: false)
        let retryID = try #require(service.imageAttachmentSettings.requestID)
        transport.emit(configJSON(requestID: expiredID, automatic: "0"))
        #expect(service.imageAttachmentSettings.requestID == retryID)
        #expect(service.runtimeConfigParams.first?.value == "1")
        transport.emit(configJSON(requestID: retryID)) // correlated response, but unchanged value
        #expect(service.imageAttachmentSettings.error != nil)
        transport.emit(configJSON())
        #expect(service.imageAttachmentSettings.error == nil)
        service.disconnect()
    }

    @Test func sendFailureAndDisconnectEndPendingChanges() async throws {
        let (service, transport) = await subject()
        transport.emit(configJSON())
        transport.rejectSends = true
        service.updateImageAttachmentSetting(.related, enabled: false)
        await drainTasks()
        #expect(service.imageAttachmentSettings.pendingSetting == nil)
        #expect(service.imageAttachmentSettings.error != nil)
        transport.rejectSends = false
        service.updateImageAttachmentSetting(.automatic, enabled: false)
        service.disconnect()
        #expect(service.imageAttachmentSettings.pendingSetting == nil)
        #expect(service.imageAttachmentSettings.error != nil)
    }

    @Test func oldServersAndDisconnectedClientsCannotUpdateImages() async {
        let (service, transport) = await subject()
        transport.emit("{\"type\":\"config_response\",\"params\":[]}")
        #expect(service.hasLoadedRuntimeConfig)
        #expect(!service.supportsImageAttachmentSettings)
        service.updateImageAttachmentSetting(.automatic, enabled: false)
        #expect(service.imageAttachmentSettings.pendingSetting == nil)
        transport.emit(configJSON())
        service.disconnect()
        service.updateImageAttachmentSetting(.automatic, enabled: false)
        #expect(service.imageAttachmentSettings.pendingSetting == nil)
    }
}

private let registeredJSON = """
{"type":"registered","device_id":"device","is_default":true,"pending_count":0}
"""

private func configJSON(requestID: String? = nil, automatic: String = "1", error: String? = nil) -> String {
    let params = ImageAttachmentSetting.allCases.map { setting in
        """
        {"key":"\(setting.rawValue)","value":"\(setting == .automatic ? automatic : "1")",
        "default":"1","description":"Image setting","type":"int","group":"iOS Attachments"}
        """
    }.joined(separator: ",")
    let correlation = requestID.map { ",\"request_id\":\"\($0)\"" } ?? ""
    let failure = error.map { ",\"error\":\"\($0)\"" } ?? ""
    return "{\"type\":\"config_response\",\"params\":[\(params)]\(correlation)\(failure)}"
}

@MainActor
private func subject() async -> (PennyService, ImageSettingsTransport) {
    let transport = ImageSettingsTransport()
    let service = PennyService(databaseService: configuredDatabase(), prefs: configuredPrefs(), webSocketClient: transport)
    await service.connect()
    transport.emit(registeredJSON)
    await drainTasks()
    return (service, transport)
}

@MainActor
private func drainTasks() async {
    for _ in 0..<30 { await Task.yield() }
}

@MainActor
private final class ImageSettingsTransport: WebSocketTransport {
    var isConnected = false
    var rejectSends = false
    var payloads: [[String: JSONValue]] = []
    private var receiver: ReceiveHandler?

    func connect(request: URLRequest, onReceive: @escaping ReceiveHandler, onFailure: @escaping FailureHandler) {
        receiver = onReceive
        isConnected = true
    }

    func disconnect() {
        isConnected = false
        receiver = nil
    }

    func send(_ data: Data) async throws {
        if rejectSends { throw URLError(.networkConnectionLost) }
        payloads.append(try JSONDecoder().decode([String: JSONValue].self, from: data))
    }

    func emit(_ json: String) { receiver?(Data(json.utf8)) }
}
