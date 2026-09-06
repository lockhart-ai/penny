import Foundation
import Observation
import PennyServices

@MainActor
@Observable
final class SettingsViewModel {
    let client: PennyWebSocketClient
    private let prefs: Prefs
    var webSocketURL: String
    var username: String
    var password: String
    var editedConfigValues: [String: String] = [:]
    var domainDraft = ""
    var domainPermission: DomainPermission = .allowed

    init(client: PennyWebSocketClient, prefs: Prefs) {
        self.client = client
        self.prefs = prefs
        webSocketURL = prefs.webSocketURL ?? ""
        username = prefs.username ?? ""
        password = prefs.password ?? ""
    }

    var canSaveConnection: Bool {
        !webSocketURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var canSendTestNotification: Bool {
        client.canSend
    }

    var apnsHost: String {
        client.apnsHost
    }

    var runtimeConfigParams: [RuntimeConfigParam] {
        client.runtimeConfigParams.filter { ImageAttachmentSetting(rawValue: $0.key) == nil }
    }

    var domainPermissions: [DomainPermissionEntry] {
        client.domainPermissions
    }

    var permissionPrompt: PermissionPrompt? {
        client.permissionPrompt
    }

    var canSubmitDomain: Bool {
        !domainDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func refresh() {
        client.requestConfig()
    }

    func configValue(for param: RuntimeConfigParam) -> String {
        editedConfigValues[param.key] ?? param.value
    }

    func setConfigValue(_ value: String, for param: RuntimeConfigParam) {
        editedConfigValues[param.key] = value
    }

    func saveConfigValue(for param: RuntimeConfigParam) {
        let value = configValue(for: param).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }
        editedConfigValues[param.key] = value
        client.updateConfig(key: param.key, value: value)
    }

    func saveConnection() {
        guard canSaveConnection else { return }
        prefs.webSocketURL = webSocketURL.trimmingCharacters(in: .whitespacesAndNewlines)
        prefs.username = username.trimmingCharacters(in: .whitespacesAndNewlines)
        prefs.password = password
        client.reconnect()
    }

    func sendTestNotification() {
        client.sendTestNotification()
    }

    func submitDomainPermission() {
        let domain = domainDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !domain.isEmpty else { return }
        client.updateDomain(domain: domain, permission: domainPermission)
        domainDraft = ""
    }

    func deleteDomain(_ entry: DomainPermissionEntry) {
        client.deleteDomain(domain: entry.domain)
    }

    func startHistorySync(channelTypes: [String], includeAttachments: Bool) {
        client.startHistorySync(
            channelTypes: channelTypes,
            includeAttachments: includeAttachments
        )
    }

    func deleteAllMessages() {
        client.deleteAllMessages()
    }

    func decidePermissionPrompt(allowed: Bool) {
        guard let prompt = permissionPrompt else { return }
        client.decidePermission(requestID: prompt.requestID, allowed: allowed)
    }
}

private func trimmedNilIfEmpty(_ value: String) -> String? {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
}
