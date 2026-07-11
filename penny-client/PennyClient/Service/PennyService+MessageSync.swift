import Foundation
import SwiftUI
import UserNotifications

extension PennyService {
    func replaceOptimisticMessage(localID: Int, with message: ChatMessage) {
        if let index = messages.firstIndex(where: { $0.id == localID }) {
            messages[index] = message
            messages.sort {
                $0.createdAt < $1.createdAt || ($0.createdAt == $1.createdAt && $0.id < $1.id)
            }
        }

        guard var displayedMessages = liveMessages?.wrappedValue,
              let index = displayedMessages.firstIndex(where: { $0.id == localID }) else { return }
        displayedMessages[index] = message
        displayedMessages.sort {
            $0.createdAt < $1.createdAt || ($0.createdAt == $1.createdAt && $0.id < $1.id)
        }
        liveMessages?.wrappedValue = displayedMessages
    }

    func clearAppBadge() {
        UNUserNotificationCenter.current().setBadgeCount(0) { error in
            if let error {
                OSLogService(category: .pennyService).error("Failed to clear badge count: \(error.localizedDescription)", privacy: .public)
            }
        }
    }
}
