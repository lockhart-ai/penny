import SwiftUI
import Testing
import UIKit
import PennyServices
@testable import PennyClient

@MainActor
struct ChatMessageAttachmentLayoutTests {
    private let digest = """
    **Collection digest**

    A few updates from your collection, with the details kept together below the image.

    **1. A new release**
    The latest update adds a refreshed design and several useful improvements. Read the \
    [release notes](https://example.com/release) for the full details.

    **2. Items to revisit**
    - Compare the available editions before choosing.
    - Check the saved notes for background and related links.
    - Keep an eye on the next update for availability.

    | Item | Status |
    | --- | --- |
    | First entry | Available |
    | Second entry | Coming soon |

    **Final note**
    This last paragraph should remain visible even when the digest spans more than a screen.
    """

    @Test func attachmentKeepsFullDigestAndLargeText() throws {
        let textOnly = try render(content: digest, attachments: [], width: 320)
        let illustrated = try render(content: digest, width: 320)
        let largeText = try render(content: digest, width: 320, textSize: .accessibility3)
        let wide = try render(content: digest, width: 600)

        // Adding an image must add height rather than compress or truncate the digest.
        #expect(illustrated.size.height > textOnly.size.height + 150)
        #expect(illustrated.size.width == 320)
        let headerPixel = try pixel(illustrated, x: 160, y: 40)
        let imagePixel = try pixel(image(), x: 240, y: 40)
        #expect(headerPixel == imagePixel)
        #expect(largeText.size.height > illustrated.size.height)
        #expect(wide.size.height < illustrated.size.height)
        try save(illustrated, name: "digest")
        try save(largeText, name: "large-text")
    }

    @Test func portraitHeaderIsBoundedAndImageOnlyHasNoCaptionPadding() throws {
        let portrait = image(size: CGSize(width: 100, height: 800))
        let imageOnly = try render(content: "", attachments: [portrait], width: 320)
        let whitespace = try render(content: " \n ", attachments: [portrait], width: 320)
        let captioned = try render(content: "A short caption.", attachments: [portrait], width: 320)

        // Source and timestamp fit in the remaining space above and below the 280pt header.
        #expect(imageOnly.size.height < 340)
        #expect(imageOnly.size.height == whitespace.size.height)
        #expect(captioned.size.height > imageOnly.size.height)
        try save(captioned, name: "portrait")
    }

    @Test func multipleAttachmentsAndOutgoingCaptionsRemainVisible() throws {
        let single = try render(content: "A short caption.", width: 320, isOutgoing: true)
        let multiple = try render(content: "A short caption.", attachments: [image(), image()], width: 320, isOutgoing: true)
        let outgoingDigest = try render(content: digest, width: 320, isOutgoing: true)

        #expect(multiple.size.height > single.size.height + 100)
        #expect(outgoingDigest.size.height > single.size.height + 200)
        try save(single, name: "outgoing")
    }

    private func image(size: CGSize = CGSize(width: 480, height: 270)) -> UIImage {
        UIGraphicsImageRenderer(size: size).image { context in
            UIColor.systemTeal.setFill()
            context.fill(CGRect(origin: .zero, size: size))
            UIColor.systemIndigo.setFill()
            context.fill(CGRect(x: 0, y: 0, width: size.width / 3, height: size.height))
        }
    }

    private func render(
        content: String,
        attachments: [UIImage]? = nil,
        width: CGFloat,
        textSize: DynamicTypeSize = .large,
        isOutgoing: Bool = false
    ) throws -> UIImage {
        let message = ChatMessage(
            id: 1, serverID: nil, createdAt: Date(timeIntervalSince1970: 0),
            content: content, sourceHint: "Sample collection",
            imageAttachments: (attachments ?? [image()]).map(ImageAttachment.init(image:)),
            isOutgoing: isOutgoing
        )
        let renderer = ImageRenderer(content: ChatMessageView(message: message, layout: .message)
            .environment(\.dynamicTypeSize, textSize)
            .environment(\.colorScheme, .light))
        renderer.proposedSize = ProposedViewSize(width: width, height: nil)
        return try #require(renderer.uiImage)
    }

    private func pixel(_ image: UIImage, x positionX: Int, y positionY: Int) throws -> [UInt8] {
        let sample = CGRect(x: CGFloat(positionX) * image.scale, y: CGFloat(positionY) * image.scale, width: 1, height: 1)
        let source = try #require(image.cgImage?.cropping(to: sample))
        var bytes = [UInt8](repeating: 0, count: 4)
        let context = try #require(CGContext(
            data: &bytes, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        context.draw(source, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        return bytes
    }

    private func save(_ image: UIImage, name: String) throws {
        let data = try #require(image.pngData())
        try data.write(to: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("attachment-\(name).png"))
    }
}
