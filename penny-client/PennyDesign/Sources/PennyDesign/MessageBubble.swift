import SwiftUI
import UIKit

/// The one chat bubble: Penny's is solid white with a hairline and a soft
/// shadow at the leading margin, the user's is solid blue at the trailing
/// margin — no avatar beside either (her avatar lives in the tab capsule).
/// An image rides the top of the bubble edge to edge, clipped to its
/// corners; the text sits below with normal padding and never renders
/// larger than body. The metadata line sits under every bubble, 4pt below
/// (`Tokens.bubbleMetaGap`). The quote block is one-way: only a user
/// bubble shows one (`UserQuoteBlock`) — Penny's never do.
///
/// The content is handed in pre-rendered (`content`) — markdown, tables
/// and links are the caller's parsers; the bubble owns the chrome. A table
/// wider than the bubble scrolls horizontally inside it: the caller marks
/// the bubble `fillsWidth` and puts the table in its own horizontal
/// scroll view. `nil` content renders an image-only bubble.
public struct MessageBubble<Content: View>: View {
    public enum Role {
        case user
        case penny
    }

    public let role: Role
    /// Images riding the top of the bubble edge to edge, in order.
    public let images: [UIImage]
    /// The quoted line shown at the top of a user's bubble; ignored on
    /// Penny's (quotes are one-way).
    public let quote: String?
    public let quoteAuthor: String
    /// Stretch the bubble to the row's full width (a table message) —
    /// otherwise a bubble without images hugs its content.
    public let fillsWidth: Bool
    // The metadata line's fields: the time, where a non-chat message came
    // from, the run's tool-call count and its formatted duration ("3 s").
    public let time: String?
    public let source: String?
    public let tools: Int?
    public let duration: String?
    public let content: Content?

    public init(
        role: Role,
        images: [UIImage] = [],
        quote: String? = nil,
        quoteAuthor: String = "Penny",
        fillsWidth: Bool = false,
        time: String? = nil,
        source: String? = nil,
        tools: Int? = nil,
        duration: String? = nil,
        @ViewBuilder content: () -> Content?
    ) {
        self.role = role
        self.images = images
        self.quote = quote
        self.quoteAuthor = quoteAuthor
        self.fillsWidth = fillsWidth
        self.time = time
        self.source = source
        self.tools = tools
        self.duration = duration
        self.content = content()
    }

    /// A bubble with images has a fixed width: the landscape flair
    /// image's, or the portrait drawing's — and a user photo's own.
    private var imageBubbleWidth: CGFloat {
        switch role {
        case .penny:
            return imageAspect < 1 ? Tokens.drawingImageWidth : Tokens.imageBubbleWidth
        case .user:
            return Tokens.userPhotoBubbleWidth
        }
    }

    /// The first image's width-over-height; the bubble width follows it.
    private var imageAspect: CGFloat {
        guard let first = images.first else { return 16.0 / 10.0 }
        return first.size.width / max(first.size.height, 1)
    }

    private var showsImages: Bool {
        !images.isEmpty
    }

    /// Height of one image in the band: its own aspect at the bubble's
    /// width, capped so an extreme aspect crops instead of towering.
    private func imageBandHeight(for image: UIImage) -> CGFloat {
        let aspect = image.size.width / max(image.size.height, 1)
        return min(imageBubbleWidth / max(aspect, 0.01), Tokens.bubbleImageMaxHeight)
    }

    public var body: some View {
        HStack(alignment: .bottom, spacing: 0) {
            if role == .user { Spacer(minLength: Tokens.bubbleSideGap) }
            VStack(alignment: role == .user ? .trailing : .leading, spacing: Tokens.bubbleMetaGap) {
                bubble
                MessageMeta(
                    role: role == .user ? .user : .penny,
                    time: time,
                    source: source,
                    tools: tools,
                    duration: duration
                )
            }
            if role == .penny { Spacer(minLength: Tokens.bubbleSideGap) }
        }
    }

    @ViewBuilder private var bubble: some View {
        if role == .penny {
            pennyBubble
        } else {
            userBubble
        }
    }

    /// Penny's bubble: white fill, 0.5pt hairline, soft shadow.
    private var pennyBubble: some View {
        VStack(alignment: .leading, spacing: 0) {
            imageBand
            if let content {
                content
                    .padding(.horizontal, Tokens.bubbleTextPaddingH)
                    .padding(.top, showsImages ? Tokens.bubbleTextTop : Tokens.bubbleTextPadding)
                    .padding(.bottom, showsImages ? Tokens.bubbleTextBottom : Tokens.bubbleTextPadding)
                    .foregroundStyle(.primary)
            }
        }
        .frame(width: showsImages ? imageBubbleWidth : nil)
        .frame(maxWidth: fillsWidth && !showsImages ? .infinity : nil, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: Tokens.bubbleCorner, style: .continuous)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(Tokens.bubbleShadowOpacity), radius: Tokens.bubbleShadowRadius, y: Tokens.bubbleShadowY)
                .overlay(
                    RoundedRectangle(cornerRadius: Tokens.bubbleCorner, style: .continuous)
                        .strokeBorder(Color(.separator).opacity(0.5), lineWidth: Tokens.bubbleHairline)
                )
        )
        .clipShape(RoundedRectangle(cornerRadius: Tokens.bubbleCorner, style: .continuous))
    }

    /// The user's bubble: solid blue. A quote rides at the top; a photo
    /// rides edge to edge above the text.
    private var userBubble: some View {
        Group {
            if showsImages {
                VStack(alignment: .leading, spacing: 0) {
                    if let quote {
                        UserQuoteBlock(author: quoteAuthor, quoted: quote)
                            .padding(.horizontal, Tokens.bubbleTextPadding)
                            .padding(.top, Tokens.bubbleTextPadding)
                    }
                    imageBand
                    if let content {
                        content
                            .foregroundStyle(.white)
                            .padding(.horizontal, Tokens.bubbleTextPaddingH)
                            .padding(.top, Tokens.bubbleTextTop)
                            .padding(.bottom, Tokens.bubbleTextPadding)
                    }
                }
                .frame(width: imageBubbleWidth)
            } else {
                VStack(alignment: .leading, spacing: Tokens.bubbleQuoteGap) {
                    if let quote {
                        UserQuoteBlock(author: quoteAuthor, quoted: quote)
                    }
                    if let content {
                        content
                            .foregroundStyle(.white)
                    }
                }
                .padding(Tokens.bubbleTextPadding)
                .frame(maxWidth: fillsWidth ? .infinity : nil, alignment: .leading)
            }
        }
        .background(Color.accentColor, in: RoundedRectangle(cornerRadius: Tokens.bubbleCorner, style: .continuous))
        .clipShape(RoundedRectangle(cornerRadius: Tokens.bubbleCorner, style: .continuous))
    }

    /// The image band: each image filling the bubble's width edge to edge,
    /// cropped to the band height — the bubble's own clipShape rounds the
    /// band's outer corners.
    @ViewBuilder private var imageBand: some View {
        ForEach(Array(images.enumerated()), id: \.offset) { _, image in
            Image(uiImage: image)
                .resizable()
                .scaledToFill()
                .frame(width: imageBubbleWidth, height: imageBandHeight(for: image))
                .clipped()
                .accessibilityHidden(true)
        }
    }
}

/// The metadata line, on every bubble: `time • source • 🛠 N • duration`,
/// leading for Penny and trailing for the user (the bubble's own VStack
/// sets the alignment). `source` names where a non-chat message came from
/// for Penny (a collection digest, a notification) and, for the user, a
/// message's own channel when it arrived from another device (#3143, "via
/// Signal") — in both cases just the second bullet segment, the one shared
/// rendering path below. `tools`/`duration` are Penny-only concepts (the
/// user's `ChatMessage` never carries either); segments are omitted when
/// absent.
public struct MessageMeta: View {
    public enum Role {
        case user
        case penny
    }

    /// The segment separator.
    private static let bullet = " • "

    public let role: Role
    public let time: String?
    public let source: String?
    public let tools: Int?
    public let duration: String?

    public init(role: Role, time: String?, source: String? = nil, tools: Int? = nil, duration: String? = nil) {
        self.role = role
        self.time = time
        self.source = source
        self.tools = tools
        self.duration = duration
    }

    public var body: some View {
        if let time {
            HStack(spacing: 0) {
                Text(time)
                if let source, !source.isEmpty {
                    Text(Self.bullet)
                    Text(source)
                }
                if role == .penny, let tools, tools > 0 {
                    Text(Self.bullet)
                    Image(systemName: "wrench.and.screwdriver")
                        .font(.caption2)
                        .accessibilityLabel("\(tools) tool calls")
                    Text(" \(tools)")
                }
                if role == .penny, let duration, !duration.isEmpty {
                    Text(Self.bullet)
                    Text(duration)
                }
            }
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
        }
    }
}

/// A quote block at the top of a user's blue bubble: a slightly lighter
/// blue fill, a 3pt white leading bar, the author bold caption, then the
/// quoted line. Only user bubbles quote; Penny's never do.
public struct UserQuoteBlock: View {
    public let author: String
    public let quoted: String

    public init(author: String, quoted: String) {
        self.author = author
        self.quoted = quoted
    }

    public var body: some View {
        HStack(alignment: .top, spacing: Tokens.quoteBarTextGap) {
            RoundedRectangle(cornerRadius: Tokens.quoteBarCorner, style: .continuous)
                .fill(Color.white)
                .frame(width: Tokens.quoteBarWidth)
            VStack(alignment: .leading, spacing: Tokens.quoteLineGap) {
                Text(author)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.white)
                Text(quoted)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.85))
                    .lineLimit(1)
            }
        }
        .padding(Tokens.quoteBlockPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.2), in: RoundedRectangle(cornerRadius: Tokens.quoteBlockCorner, style: .continuous))
    }
}
