import SwiftUI
import UIKit
import UserNotifications

@main struct MyApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    static var isRunningTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }

    var body: some Scene {
        WindowGroup {
            if Self.isRunningTests {
                Color.clear
            } else {
                MessageView()
            }
        }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {
    private let logger = OSLogService(category: .notifications)

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        guard !MyApp.isRunningTests else { return true }

        let notificationCenter = UNUserNotificationCenter.current()
        notificationCenter.delegate = self
        logNotificationSettings(using: notificationCenter)
        requestNotificationRegistration(application, using: notificationCenter)
        return true
    }

    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        let token = deviceToken.map { String(format: "%02x", $0) }.joined()
        PushNotificationState.shared.updateDeviceToken(token)
    }

    func application(_ application: UIApplication, didFailToRegisterForRemoteNotificationsWithError error: Error) {
        logger.error("Failed to register for remote notifications: \(error.localizedDescription)", privacy: .public)
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        applyBadge(from: notification, using: center)
        return [.banner, .list, .sound, .badge]
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse
    ) async {
        logger.info("Received notification response: \(response.notification.request.identifier)", privacy: .public)
        applyBadge(from: response.notification, using: center)
    }

    private func applyBadge(from notification: UNNotification, using center: UNUserNotificationCenter) {
        guard let badge = notification.request.content.badge else {
            logger.warning("Notification did not include an aps.badge value")
            return
        }

        let badgeCount = badge.intValue
        logger.info("Applying notification badge count: \(badgeCount)", privacy: .public)
        center.setBadgeCount(badgeCount) { error in
            if let error {
                self.logger.error("Failed to set badge count: \(error.localizedDescription)", privacy: .public)
            }
        }
    }

    private func requestNotificationRegistration(_ application: UIApplication, using center: UNUserNotificationCenter) {
        center.requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            if let error {
                self.logger.error("Notification authorization failed: \(error.localizedDescription)", privacy: .public)
                return
            }

            self.logNotificationSettings(using: center)
            guard granted else { return }

            DispatchQueue.main.async {
                application.registerForRemoteNotifications()
            }
        }
    }

    private func logNotificationSettings(using center: UNUserNotificationCenter) {
        center.getNotificationSettings { settings in
            self.logger.info("Notification authorization: \(settings.authorizationStatus.rawValue), badge setting: \(settings.badgeSetting.rawValue)", privacy: .public)
        }
    }
}

@MainActor
final class PushNotificationState {
    static let didUpdateDeviceToken = Notification.Name("PushNotificationState.didUpdateDeviceToken")
    static let shared = PushNotificationState()

    private(set) var deviceToken: String?

    func updateDeviceToken(_ token: String) {
        guard deviceToken != token else { return }
        deviceToken = token
        NotificationCenter.default.post(name: Self.didUpdateDeviceToken, object: nil)
    }
}
