import Observation
import SwiftUI
import UIKit

extension MessageView {
    @MainActor
    @Observable
    final class ViewModel {
        var client = PennyWebSocketClient()
        var draftMessage = ""
        var isShowingConnectionError = false
        var isShowingSettings = false
        var composerHeight: CGFloat = 64
        var keyboardHeight: CGFloat = 0

        init(client: PennyWebSocketClient? = nil) {
            self.client = client ?? PennyWebSocketClient()
        }

        private let keyboardComposerSpacing: CGFloat = -24

        var keyboardOffset: CGFloat {
            keyboardHeight > 0 ? keyboardHeight + keyboardComposerSpacing : 0
        }

        func connect() async {
            await client.connect()
        }

        func disconnect() {
            client.disconnect()
        }

        func reconnect() {
            client.reconnect()
        }

        func sendDraft() {
            let trimmedMessage = draftMessage.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedMessage.isEmpty else { return }

            draftMessage = ""
            client.sendMessage(trimmedMessage)
        }

        func handleScenePhaseChange(_ phase: ScenePhase) {
            switch phase {
            case .active:
                Task { await client.connect() }
            case .background:
                client.disconnect()
            case .inactive:
                break
            @unknown default:
                break
            }
        }

        func updateKeyboardHeight(from notification: Notification) {
            guard let keyboardFrame = notification.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect else { return }

            let screenHeight = UIApplication.shared.connectedScenes
                .compactMap { ($0 as? UIWindowScene)?.screen.bounds.height }
                .first ?? keyboardFrame.maxY
            let overlap = max(0, screenHeight - keyboardFrame.minY)
            let duration = notification.userInfo?[UIResponder.keyboardAnimationDurationUserInfoKey] as? Double ?? 0.25

            withAnimation(.easeOut(duration: duration)) {
                keyboardHeight = overlap
            }
        }
    }
}
