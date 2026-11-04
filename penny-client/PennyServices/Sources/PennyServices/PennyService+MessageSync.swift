import Foundation
import SwiftUI
import UserNotifications

extension PennyService {
    /// History pages can enrich messages already on screen without replacing their attachments.
    func updateRunLinks(_ incoming: [ChatMessage]) {
        let received = Dictionary(uniqueKeysWithValues: incoming.map { ($0.id, $0) })
        func update(_ stored: inout [ChatMessage]) -> Bool {
            var changed = false
            for index in stored.indices {
                guard let message = received[stored[index].id], message.runID != stored[index].runID else { continue }
                stored[index].runID = message.runID
                changed = true
            }
            return changed
        }
        _ = update(&messages)
        if var displayed = liveMessages?.wrappedValue, update(&displayed) {
            liveMessages?.wrappedValue = displayed
        }
    }

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
        // The standalone service test bundle has no application bundle proxy,
        // so UserNotifications is unavailable there.
        guard ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil else { return }
        UNUserNotificationCenter.current().setBadgeCount(0) { error in
            if let error {
                OSLogService(category: .pennyService).error("Failed to clear badge count: \(error.localizedDescription)", privacy: .public)
            }
        }
    }
}
