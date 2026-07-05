import SwiftUI

struct ChatMessageView: View {
    let message: ChatMessage
    let layout: MessageView.MessageLayout
    let showsSourceHintInline: Bool
    let fillsMessageRowWidth: Bool

    init(message: ChatMessage, layout: MessageView.MessageLayout, showsSourceHintInline: Bool = true, fillsMessageRowWidth: Bool = false) {
        self.message = message
        self.layout = layout
        self.showsSourceHintInline = showsSourceHintInline
        self.fillsMessageRowWidth = fillsMessageRowWidth
    }

    private var markdownTextBlocks: [AttributedString] {
        message.content
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { line in
                let text = line.isEmpty ? " " : String(line)
                return (try? AttributedString(markdown: text)) ?? AttributedString(text)
            }
    }

    private var sourceTitle: String {
        if let sourceHint = message.sourceHint, !sourceHint.isEmpty {
            return sourceHint
        }
        return message.isOutgoing ? "You" : "Penny"
    }

    private var firstAttachment: ImageAttachment? {
        message.imageAttachments.first
    }

    private var compactCardBackground: Color {
        message.isOutgoing ? Color.accentColor : Color(.secondarySystemGroupedBackground)
    }

    private var compactPrimaryForeground: Color {
        message.isOutgoing ? .white : .primary
    }

    private var compactSecondaryForeground: Color {
        message.isOutgoing ? .white.opacity(0.78) : .secondary
    }

    private func attachmentThumbnail(_ attachment: ImageAttachment, height: CGFloat, cornerRadius: CGFloat) -> some View {
        GeometryReader { proxy in
            Image(uiImage: attachment.image)
                .resizable()
                .scaledToFill()
                .frame(width: proxy.size.width, height: height)
                .clipped()
        }
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }

    @ViewBuilder
    private var messageBubble: some View {
        if message.isOutgoing {
            Text(message.content)
                .font(.body)
                .foregroundStyle(.white)
                .frame(maxWidth: fillsMessageRowWidth ? .infinity : nil, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        } else {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(markdownTextBlocks.indices, id: \.self) { index in
                    Text(markdownTextBlocks[index])
                        .lineLimit(nil)
                        .font(.body)
                }

                ForEach(message.imageAttachments) { attachment in
                    Image(uiImage: attachment.image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: 260)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
            }
            .foregroundStyle(.primary)
            .frame(maxWidth: fillsMessageRowWidth ? .infinity : nil, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
    }

    var body: some View {
        switch layout {
        case .message:
            messageRow
        case .compact:
            compactCard
        case .media:
            mediaCard
        }
    }

    private var messageRow: some View {
        HStack {
            if message.isOutgoing && !fillsMessageRowWidth {
                Spacer(minLength: 48)
            }

            VStack(alignment: message.isOutgoing && !fillsMessageRowWidth ? .trailing : .leading, spacing: 5) {
                if showsSourceHintInline, let sourceHint = message.sourceHint, !sourceHint.isEmpty {
                    Text(sourceHint)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }

                messageBubble

                Text(message.displayTime)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if !message.isOutgoing && !fillsMessageRowWidth {
                Spacer(minLength: 48)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var compactCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let firstAttachment {
                attachmentThumbnail(firstAttachment, height: 96, cornerRadius: 8)
            } else if message.isOutgoing {
                Color.clear
                    .frame(maxWidth: .infinity)
                    .frame(height: 96)
            }

            Text(sourceTitle)
                .font(.caption.weight(.semibold))
                .foregroundStyle(compactSecondaryForeground)
                .lineLimit(1)
                .frame(height: 14, alignment: .top)

            Text(message.content)
                .font(.subheadline)
                .foregroundStyle(compactPrimaryForeground)
                .lineLimit(3)
                .frame(maxWidth: .infinity, minHeight: 54, alignment: .topLeading)
        }
        .padding(8)
        .frame(maxWidth: .infinity, minHeight: 192, alignment: .topLeading)
        .background(compactCardBackground, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var mediaCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            mediaBubble

            Text(sourceTitle)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private var mediaBubble: some View {
        Group {
            if let firstAttachment {
                attachmentThumbnail(firstAttachment, height: 86, cornerRadius: 8)
            } else {
                Image(systemName: "photo")
                    .font(.title2)
                    .foregroundStyle(message.isOutgoing ? .white.opacity(0.78) : Color(.tertiaryLabel))
                    .frame(maxWidth: .infinity)
                    .frame(height: 86)
                    .background(message.isOutgoing ? Color.accentColor : Color(.tertiarySystemGroupedBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
        }
        .padding(message.isOutgoing ? 3 : 0)
        .background(message.isOutgoing ? Color.accentColor : Color.clear, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

extension MessageView.MessageLayout {
    var columnCount: Int {
        switch self {
        case .message:
            return 1
        case .compact:
            return 2
        case .media:
            return 3
        }
    }

    var horizontalPadding: CGFloat {
        switch self {
        case .message:
            return 16
        case .compact:
            return 10
        case .media:
            return 8
        }
    }

    var itemSpacing: CGFloat {
        switch self {
        case .message:
            return 12
        case .compact:
            return 8
        case .media:
            return 6
        }
    }
}
