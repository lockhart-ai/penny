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

    /// Saves on `.onSubmit`/focus loss rather than a Save button (#3169), so
    /// this fires on every field visit, not just an edit — skip the no-op
    /// write (and its `config_update`) when the trimmed value already
    /// matches the server's.
    func saveConfigValue(for param: RuntimeConfigParam) {
        let value = configValue(for: param).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, value != param.value else { return }
        editedConfigValues[param.key] = value
        client.updateConfig(key: param.key, value: value)
    }

    /// Saves on `.onSubmit`/focus loss rather than a Save button (#3169).
    /// Reconnects once, only when a saved value actually changed — tabbing
    /// through the Connection fields without editing them must not
    /// reconnect at all, and editing one field must not reconnect twice
    /// (once from `.onSubmit`, again when focus then leaves the field).
    func saveConnection() {
        guard canSaveConnection else { return }
        let trimmedURL = webSocketURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedUsername = username.trimmingCharacters(in: .whitespacesAndNewlines)
        let changed = trimmedURL != (prefs.webSocketURL ?? "")
            || trimmedUsername != (prefs.username ?? "")
            || password != (prefs.password ?? "")
        prefs.webSocketURL = trimmedURL
        prefs.username = trimmedUsername
        prefs.password = password
        guard changed else { return }
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
