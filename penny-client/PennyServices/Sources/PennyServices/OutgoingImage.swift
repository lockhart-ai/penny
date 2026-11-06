import UIKit

/// The outgoing photo encoder: a photo picked in the composer leaves the
/// client as a JPEG data URI on the `message` frame's optional `images`
/// field — downscaled so its long edge is at most `maxLongEdge` pixels and
/// re-encoded at `jpegQuality`, so a full-resolution library photo never
/// ships megabytes over the websocket.
enum OutgoingImage {
    /// The long edge's pixel cap before re-encoding.
    static let maxLongEdge: CGFloat = 2048
    /// The JPEG compression quality of every outgoing photo.
    static let jpegQuality: CGFloat = 0.8
    /// The data URI's media type — everything leaves as JPEG.
    static let dataURLMediaType = "image/jpeg"

    /// The photo as a `data:image/jpeg;base64,…` data URI, downscaled and
    /// re-encoded; nil when the image cannot be encoded at all.
    static func dataURL(for image: UIImage) -> String? {
        guard let jpegData = jpegData(for: image) else { return nil }
        return "data:\(dataURLMediaType);base64,\(jpegData.base64EncodedString())"
    }

    static func jpegData(for image: UIImage) -> Data? {
        downscaled(image).jpegData(compressionQuality: jpegQuality)
    }

    /// The image at most `maxLongEdge` pixels on its long edge, keeping its
    /// aspect and orientation. Smaller images pass through untouched.
    private static func downscaled(_ image: UIImage) -> UIImage {
        let pixelWidth = image.size.width * image.scale
        let pixelHeight = image.size.height * image.scale
        let longEdge = max(pixelWidth, pixelHeight)
        guard longEdge > maxLongEdge, longEdge > 0 else { return image }

        let scale = maxLongEdge / longEdge
        let targetSize = CGSize(width: pixelWidth * scale, height: pixelHeight * scale)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}
