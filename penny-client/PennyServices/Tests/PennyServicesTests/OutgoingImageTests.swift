import Foundation
import Testing
import UIKit
@testable import PennyServices

/// The outgoing photo path (#3039): the composer's picked photo leaves as a
/// JPEG data URI on the `message` frame's optional `images` field —
/// downscaled to the long-edge cap — and a server error frame while a photo
/// message is unanswered posts an honest notice into the chat.
@Suite(.serialized)
@MainActor
struct OutgoingImageTests {
    @Test func encoderCapsLongEdgeAndKeepsAspect() throws {
        let wide = makeImage(width: 3000, height: 1500)
        let jpeg = try #require(OutgoingImage.jpegData(for: wide))
        let encoded = try #require(UIImage(data: jpeg))
        #expect(max(encoded.size.width, encoded.size.height) <= OutgoingImage.maxLongEdge)
        #expect(encoded.size.width / encoded.size.height == 2)

        let small = makeImage(width: 800, height: 500)
        let smallJpeg = try #require(OutgoingImage.jpegData(for: small))
        let untouched = try #require(UIImage(data: smallJpeg))
        #expect(untouched.size.width == 800)
        #expect(untouched.size.height == 500)
    }

    @Test func encoderProducesJpegDataURL() throws {
        let dataURL = try #require(OutgoingImage.dataURL(for: makeImage(width: 100, height: 100)))
        #expect(dataURL.hasPrefix("data:\(OutgoingImage.dataURLMediaType);base64,"))

        let base64 = String(dataURL.dropFirst("data:\(OutgoingImage.dataURLMediaType);base64,".count))
        let data = try #require(Data(base64Encoded: base64))
        #expect(UIImage(data: data) != nil)
    }

    @Test func sendMessageCarriesImagesOnTheMessageFrame() async throws {
        let (service, transport) = await connectedService()
        service.sendMessage("What's this bird?", images: [makeImage(width: 200, height: 120)])
        await drainTasks()

        let payload = try #require(transport.payloads.last)
        #expect(payload["type"] == .string("message"))
        #expect(payload["content"] == .string("What's this bird?"))
        guard case .array(let images)? = payload["images"] else {
            Issue.record("message frame carried no images")
            return
        }
        #expect(images.count == 1)
        guard case .string(let dataURL)? = images.first else {
            Issue.record("image entry was not a data URI string")
            return
        }
        #expect(dataURL.hasPrefix("data:\(OutgoingImage.dataURLMediaType);base64,"))
    }

    @Test func sendMessageSendsImageOnlyMessages() async throws {
        let (service, transport) = await connectedService()
        service.sendMessage("", images: [makeImage(width: 200, height: 120)])
        await drainTasks()

        let payload = try #require(transport.payloads.last)
        #expect(payload["type"] == .string("message"))
        #expect(payload["content"] == .string(""))
        guard case .array(let images)? = payload["images"] else {
            Issue.record("message frame carried no images")
            return
        }
        #expect(images.count == 1)
    }

    @Test func sendMessageWithoutImagesOmitsTheField() async throws {
        let (service, transport) = await connectedService()
        service.sendMessage("Plain text")
        await drainTasks()

        let payload = try #require(transport.payloads.last)
        #expect(payload["type"] == .string("message"))
        #expect(payload["images"] == nil)
    }

    @Test func serverErrorAfterImageSendPostsChatNotice() async throws {
        let (service, transport) = await connectedService()
        service.sendMessage("What's this bird?", images: [makeImage(width: 200, height: 120)])
        let sentCount = service.messages.count
        transport.emit(statusJSON(error: "Image too large"))
        await drainTasks()

        let notice = try #require(service.messages.last)
        #expect(notice.isOutgoing == false)
        #expect(notice.content == PennyService.imageDeliveryFailureNoticePrefix + "Image too large")
        #expect(service.messages.count == sentCount + 1)
    }

    @Test func serverErrorWithoutPendingImageSendStaysOutOfTheChat() async throws {
        let (service, transport) = await connectedService()
        service.sendMessage("Plain text")
        let sentCount = service.messages.count
        transport.emit(statusJSON(error: "Something broke"))
        await drainTasks()

        #expect(service.messages.count == sentCount)
        #expect(service.lastError == "Something broke")
    }

    @Test func oneNoticePerPendingImageSend() async throws {
        let (service, transport) = await connectedService()
        service.sendMessage("", images: [makeImage(width: 200, height: 120)])
        transport.emit(statusJSON(error: "First failure"))
        transport.emit(statusJSON(error: "Second failure"))
        await drainTasks()

        let notices = service.messages.filter { $0.content.hasPrefix(PennyService.imageDeliveryFailureNoticePrefix) }
        #expect(notices.count == 1)
    }

    // MARK: Helpers

    private func makeImage(width: CGFloat, height: CGFloat) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
        return renderer.image { context in
            UIColor.systemBlue.setFill()
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        }
    }

    private func connectedService() async -> (PennyService, CapturingTransport) {
        let transport = CapturingTransport()
        let service = PennyService(databaseService: configuredDatabase(), prefs: configuredPrefs(), webSocketClient: transport)
        await service.connect()
        transport.emit(registeredJSON)
        await drainTasks()
        return (service, transport)
    }

    private func drainTasks() async {
        for _ in 0..<30 { await Task.yield() }
    }

    private func statusJSON(error: String) -> String {
        "{\"type\":\"status\",\"connected\":true,\"error\":\"\(error)\"}"
    }
}

private let registeredJSON = """
{"type":"registered","device_id":"device","is_default":true,"pending_count":0}
"""

@MainActor
private final class CapturingTransport: WebSocketTransport {
    var isConnected = false
    var payloads: [[String: JSONValue]] = []
    private var receiver: ReceiveHandler?

    func connect(request: URLRequest, onReceive: @escaping ReceiveHandler, onFailure: @escaping FailureHandler) {
        receiver = onReceive
        isConnected = true
    }

    func disconnect() {
        isConnected = false
        receiver = nil
    }

    func send(_ data: Data) async throws {
        payloads.append(try JSONDecoder().decode([String: JSONValue].self, from: data))
    }

    func emit(_ json: String) {
        receiver?(Data(json.utf8))
    }
}
