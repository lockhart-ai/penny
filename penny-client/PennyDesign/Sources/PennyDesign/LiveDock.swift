import SwiftUI

/// One entry of Penny's live work stream (#3042): a thinking chunk or a
/// tool call, chronological, oldest first. `id` is derived from the run's
/// structural position (run id + step number + kind) by the caller, never
/// a random UUID — an unchanged earlier item keeps its SwiftUI identity as
/// later ones are appended, instead of the whole stream re-identifying
/// itself on every progress event.
public struct LiveDockItem: Identifiable {
    public enum Kind {
        /// A thinking chunk: italic footnote text with a sparkle.
        case thinking(String)
        /// A tool call: its icon, a short label, and whether it is still
        /// running (the tools of the run's last step) or done.
        case tool(symbol: String, label: String, running: Bool)
    }

    public let id: String
    public let kind: Kind

    public init(id: String, kind: Kind) {
        self.id = id
        self.kind = kind
    }
}

private struct LiveDockRow: View {
    let item: LiveDockItem

    var body: some View {
        switch item.kind {
        case .thinking(let text):
            IconRow(textStyle: .footnote, weight: .regular, spacing: 8) {
                Image(systemName: "sparkle")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(width: Tokens.iconColumn, alignment: .center)
                    .accessibilityHidden(true)
            } content: {
                Text(text)
                    .font(.footnote)
                    .italic()
                    .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .combine)
        case .tool(let symbol, let label, let running):
            IconRow(textStyle: .footnote, weight: .regular, spacing: 8) {
                Image(systemName: symbol)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(width: Tokens.iconColumn, alignment: .center)
                    .accessibilityHidden(true)
            } content: {
                Text(label)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                if running {
                    RunningEllipsis()
                        .accessibilityHidden(true)
                } else {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.blue)
                        .font(.footnote)
                        .accessibilityHidden(true)
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("\(label), \(running ? "in progress" : "done")")
        }
    }
}

/// The live dock (#3042): no collapsed state. A header (the app's own
/// avatar with the blue progress ring · "Working · Step n · t s", optional
/// "· n queued") over a chronological stream of thinking chunks and tool
/// rows, oldest at the top. It grows to fit its content up to 40% of the
/// screen height (`Tokens.dockScreenCapFraction`); past that the body
/// scrolls internally, pinned to the newest item, with a soft fade at the
/// top edge. It sits in the conversation's flow on rainbow-edged glass —
/// the last item, where Penny's reply will appear. The resolve morph is a
/// plain crossfade (`contentOpacity`), not a shared glass identity: two
/// views simultaneously claiming one `glassEffectID` is unsupported and
/// crashes, and the dock's content and the bubble's are different shapes
/// at different positions, not two frames of the same shape.
///
/// The avatar is handed in (`@ViewBuilder avatar`) — the same seam
/// `TopTabBar` uses for the chat tab's icon: the kit sizes and rings
/// whatever the app hands it; the app owns the actual asset.
///
/// There is no stop control here yet: cancelling a run has no message in
/// the iOS protocol today, so showing a button with nothing to send would
/// be a dead control — it lands once that wire message exists.
public struct LiveDock<Avatar: View>: View {
    public let items: [LiveDockItem]
    public let step: Int
    public let startedAt: Date
    public var queued: Int = 0
    /// Fades the dock's content during the resolve morph — the stream
    /// fading out while the arriving reply fades in through its own,
    /// separate arrival animation.
    public var contentOpacity: Double?
    @ViewBuilder public var avatar: () -> Avatar

    public init(
        items: [LiveDockItem],
        step: Int,
        startedAt: Date,
        queued: Int = 0,
        contentOpacity: Double? = nil,
        @ViewBuilder avatar: @escaping () -> Avatar
    ) {
        self.items = items
        self.step = step
        self.startedAt = startedAt
        self.queued = queued
        self.contentOpacity = contentOpacity
        self.avatar = avatar
    }

    @State private var headerHeight: CGFloat = 0
    @State private var contentHeight: CGFloat = 0

    private let bottomMarker = "live-dock-bottom"
    private let dockPadding: CGFloat = 16
    private let headerBodySpacing: CGFloat = 10

    private var cap: CGFloat { UIScreen.main.bounds.height * Tokens.dockScreenCapFraction }
    /// The stream's own ceiling: whatever the screen cap leaves after the
    /// header, the two paddings and the header→body spacing.
    private var bodyMax: CGFloat {
        max(cap - headerHeight - 2 * dockPadding - headerBodySpacing, 40)
    }
    /// Size to the content: the stream's natural height, capped at bodyMax —
    /// never the ScrollView's offered height.
    private var bodyHeight: CGFloat { min(contentHeight, bodyMax) }
    private var scrolls: Bool { contentHeight > bodyMax }

    private func headerText(elapsed: Int) -> String {
        var text = "Working · Step \(step) · \(elapsed) s"
        if queued > 0 { text += " · \(queued) queued" }
        return text
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: headerBodySpacing) {
            header
                .onGeometryChange(for: CGFloat.self, of: { $0.size.height },
                                  action: { headerHeight = $0 })
            stream
        }
        .padding(dockPadding)
        .glassEffect(.regular, in: .rect(cornerRadius: 24))
        .background {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .strokeBorder(pennyRainbow, lineWidth: 4)
                .blur(radius: 6)
                .opacity(0.5)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .strokeBorder(pennyRainbow, lineWidth: 2)
        }
    }

    /// The elapsed seconds tick off a `TimelineView` rather than a value
    /// read once — the header keeps counting while the dock is showing.
    /// One accessibility element reads the full status text regardless of
    /// the visual `lineLimit(1)` truncation; the avatar and progress ring
    /// are decorative on their own (the status text already says she's
    /// working).
    private var header: some View {
        TimelineView(.periodic(from: startedAt, by: 1)) { context in
            let text = headerText(elapsed: max(0, Int(context.date.timeIntervalSince(startedAt))))
            HStack(spacing: 12) {
                ZStack {
                    avatar()
                        .frame(width: 28, height: 28)
                    Circle()
                        .trim(from: 0, to: 0.7)
                        .stroke(Color.blue, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                        .frame(width: 34, height: 34)
                        .rotationEffect(.degrees(-90))
                }
                .frame(width: Tokens.iconColumn - 4, height: Tokens.iconColumn - 4)
                .accessibilityHidden(true)
                Text(text)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Spacer(minLength: 8)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(text)
        }
    }

    private var stream: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(items) { LiveDockRow(item: $0) }
                    Color.clear.frame(height: 1).id(bottomMarker)
                }
                .onGeometryChange(for: CGFloat.self, of: { $0.size.height },
                                  action: { contentHeight = $0 })
            }
            .frame(height: bodyHeight)
            .mask {
                if scrolls {
                    LinearGradient(stops: [
                        .init(color: .clear, location: 0),
                        .init(color: .black, location: 0.22),
                        .init(color: .black, location: 1)
                    ], startPoint: .top, endPoint: .bottom)
                } else {
                    Rectangle().fill(Color.black)
                }
            }
            .task {
                proxy.scrollTo(bottomMarker, anchor: .bottom)
            }
            .onChange(of: contentHeight) { _, _ in
                proxy.scrollTo(bottomMarker, anchor: .bottom)
            }
        }
        .opacity(contentOpacity ?? 1)
    }
}

/// The app-wide container for a leading icon beside text, kept for the
/// chat dock's stream rows — the one place a row's icon and content are
/// bespoke. Every list row in the app comes from `ListRow` instead. The
/// rule, literal: the icon's TOP EDGE is level with the TOP OF THE CAPITAL
/// LETTERS of the first line of text (the cap height) — the row aligns on
/// the first text baseline and the icon's baseline guide offsets down by
/// the first line's cap height, taken from the real font metrics.
private struct IconRow<Icon: View, Content: View>: View {
    let textStyle: Font.TextStyle
    let weight: Font.Weight
    var spacing: CGFloat = Tokens.iconTextGap
    @ViewBuilder var icon: () -> Icon
    @ViewBuilder var content: () -> Content

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: spacing) {
            icon()
                .alignmentGuide(.firstTextBaseline) { dimensions in
                    dimensions[.top] + capHeight(textStyle, weight)
                }
            content()
        }
    }
}
