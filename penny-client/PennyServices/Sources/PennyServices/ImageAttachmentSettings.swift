import Foundation
import Observation

public enum ImageAttachmentSetting: String, CaseIterable, Identifiable, Sendable {
    case automatic = "IOS_AUTOMATIC_IMAGES"
    case citedPage = "IOS_CITED_PAGE_IMAGES"
    case sameSite = "IOS_SAME_SITE_IMAGES"
    case related = "IOS_RELATED_IMAGES"

    public var id: String { rawValue }
}

/// Tracks confirmation separately from the server's authoritative config values.
@MainActor
@Observable
public final class ImageAttachmentSettings {
    public private(set) var pendingSetting: ImageAttachmentSetting?
    public private(set) var pendingValue: Bool?
    public private(set) var error: String?
    private(set) var requestID: String?
    @ObservationIgnored private var timeoutTask: Task<Void, Never>?

    func begin(_ setting: ImageAttachmentSetting, enabled: Bool) -> String {
        let identifier = UUID().uuidString
        requestID = identifier
        pendingSetting = setting
        pendingValue = enabled
        error = nil
        timeoutTask?.cancel()
        timeoutTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .seconds(10))
                self?.fail(requestID: identifier, message: "Could not confirm the change. Refresh settings and try again.")
            } catch { return }
        }
        return identifier
    }

    func receive(_ payload: ConfigResponsePayload) -> Bool {
        guard let identifier = payload.requestID else {
            if pendingSetting == nil { error = nil }
            return true
        }
        guard identifier == requestID, let setting = pendingSetting else { return false }
        let value = payload.params.first { $0.key == setting.rawValue }?.value
        let expectedValue = pendingValue == true ? "1" : "0"
        let failure = payload.error ?? (value == expectedValue ? nil : "The change was not saved. Try again.")
        finish(error: failure)
        return true
    }

    func fail(requestID identifier: String?, message: String) {
        guard let identifier, identifier == requestID else { return }
        finish(error: message)
    }

    func disconnect() {
        guard requestID != nil else { return }
        finish(error: "Connection lost before the change was confirmed. Reconnect to refresh settings.")
    }

    private func finish(error: String?) {
        timeoutTask?.cancel()
        timeoutTask = nil
        requestID = nil
        pendingSetting = nil
        pendingValue = nil
        self.error = error
    }
}
