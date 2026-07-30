import Foundation

@MainActor
public final class PushNotificationState {
    public static let didUpdateDeviceToken = Notification.Name("PushNotificationState.didUpdateDeviceToken")
    public static let shared = PushNotificationState()

    private(set) public var deviceToken: String?

    public func updateDeviceToken(_ token: String) {
        guard deviceToken != token else { return }
        deviceToken = token
        NotificationCenter.default.post(name: Self.didUpdateDeviceToken, object: nil)
    }
}
