import Foundation
import Testing
import PennyServices
@testable import PennyClient

/// The cross-device "via" caption (#3143): a user bubble whose message
/// arrived from a channel other than this app names it; one sent from this
/// app, or Penny's own reply, carries none.
struct ChatMessageViewTests {
    @Test func nonIOSChannelShowsViaCaption() {
        let message = userMessage(channelType: "signal", deviceLabel: "Signal")
        let view = ChatMessageView(message: message, layout: .message)

        #expect(view.viaCaption == "via Signal")
    }

    @Test func iOSChannelShowsNoCaption() {
        let message = userMessage(channelType: "ios", deviceLabel: "iPhone")
        let view = ChatMessageView(message: message, layout: .message)

        #expect(view.viaCaption == nil)
    }

    @Test func discordAndBrowserMapToTheirDisplayNames() {
        #expect(ChatMessageView(message: userMessage(channelType: "discord"), layout: .message).viaCaption == "via Discord")
        #expect(ChatMessageView(message: userMessage(channelType: "browser"), layout: .message).viaCaption == "via Browser")
    }

    @Test func unknownChannelFallsBackToTheRawDeviceLabel() {
        let message = userMessage(channelType: "pager", deviceLabel: "Office Pager")
        let view = ChatMessageView(message: message, layout: .message)

        #expect(view.viaCaption == "via Office Pager")
    }

    @Test func missingChannelInformationShowsNoCaption() {
        let message = userMessage(channelType: nil, deviceLabel: nil)
        let view = ChatMessageView(message: message, layout: .message)

        #expect(view.viaCaption == nil)
    }

    @Test func pennyRepliesNeverShowAViaCaption() {
        let message = ChatMessage(
            id: 1, serverID: 1, createdAt: Date(timeIntervalSince1970: 0),
            content: "Reply", sourceHint: nil, channelType: "signal", deviceLabel: "Signal",
            imageAttachments: [], isOutgoing: false
        )
        let view = ChatMessageView(message: message, layout: .message)

        #expect(view.viaCaption == nil)
    }

    private func userMessage(channelType: String?, deviceLabel: String? = nil) -> ChatMessage {
        ChatMessage(
            id: 1, serverID: 1, createdAt: Date(timeIntervalSince1970: 0),
            content: "Add a bag of dark roast too?", sourceHint: nil,
            channelType: channelType, deviceLabel: deviceLabel,
            imageAttachments: [], isOutgoing: true
        )
    }
}
