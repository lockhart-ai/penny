import SwiftUI
import UIKit
import UserNotifications
import os
import PennyServices

@main struct MyApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    static var isRunningTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }

    #if DEBUG
    /// The story named by a `-PennyStory <id>` launch argument (story mode,
    /// #2965), or nil for ordinary launches. Debug builds only — release/
    /// TestFlight builds carry no story code at all.
    static var activeStoryID: StoryID? {
        guard let flagIndex = ProcessInfo.processInfo.arguments.firstIndex(of: "-PennyStory") else { return nil }
        let valueIndex = flagIndex + 1
        guard valueIndex < ProcessInfo.processInfo.arguments.count else { return nil }
        return StoryID(rawValue: ProcessInfo.processInfo.arguments[valueIndex])
    }
    #endif

    var body: some Scene {
        WindowGroup {
            if Self.isRunningTests {
                Color.clear
            } else {
                #if DEBUG
                if let storyID = Self.activeStoryID {
                    StoryRootView(storyID: storyID)
                } else {
                    MessageView()
                }
                #else
                MessageView()
                #endif
            }
        }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {
    private let logger = Logger(subsystem: "PennyClient", category: "notifications")

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        guard !MyApp.isRunningTests else { return true }
        #if DEBUG
        guard MyApp.activeStoryID == nil else { return true }
        #endif

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
        logger.error("Failed to register for remote notifications: \(error.localizedDescription)")
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
        logger.info("Received notification response: \(response.notification.request.identifier)")
        applyBadge(from: response.notification, using: center)
    }

    private func applyBadge(from notification: UNNotification, using center: UNUserNotificationCenter) {
        guard let badge = notification.request.content.badge else {
            logger.warning("Notification did not include an aps.badge value")
            return
        }

        let badgeCount = badge.intValue
        logger.info("Applying notification badge count: \(badgeCount)")
        center.setBadgeCount(badgeCount) { error in
            if let error {
                self.logger.error("Failed to set badge count: \(error.localizedDescription)")
            }
        }
    }

    private func requestNotificationRegistration(_ application: UIApplication, using center: UNUserNotificationCenter) {
        center.requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            if let error {
                self.logger.error("Notification authorization failed: \(error.localizedDescription)")
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
            self.logger.info("Notification authorization: \(settings.authorizationStatus.rawValue), badge setting: \(settings.badgeSetting.rawValue)")
        }
    }
}
