import SwiftUI
import UIKit
import PennyDesign
import PennyServices

struct MessageView: View {
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.shellContentTopClearance) private var shellContentTopClearance
    @Environment(\.shellOpenPrompts) private var shellOpenPrompts
    @State private var viewModel: ViewModel
    @State private var presentedCardMessage: ChatMessage?
    @State private var messageFrames: [Int: CGRect] = [:]
    @State private var messageScrollFrames: [Int: CGRect] = [:]
    @State private var keyboardSettledScrollTask: Task<Void, Never>?
    @State private var arrivalAnimatedMessageIDs: Set<Int> = []
    @State private var pendingIncomingAutoScrollMessageID: Int?
    @State private var lastTopScrolledMessageID: Int?
    @State private var chatViewportFrame: CGRect = .zero
    @FocusState private var isComposerFocused: Bool

    // MARK: Live dock (#3042)
    /// The matched-geometry identity shared by the dock and the one
    /// message it resolves into, so the dock's frame interpolates into the
    /// reply bubble's frame in the same flow slot rather than the two
    /// appearing in different places. A `Namespace` only exists on a View,
    /// so this is the one piece of dock state that can't live on the view
    /// model with the rest.
    @Namespace private var dockNamespace
    private let dockGeometryID = "dock-resolve"
    /// Story mode only (#2965): freezes the dock mid-morph at a fixed
    /// opacity instead of animating it, so the transition is screenshottable.
    private let storyDockResolveProgress: Double?

    /// `presentedCardMessage` only exists so story mode (#2965) can open the
    /// card detail sheet already presented over the chat story. It defaults
    /// to today's behaviour, so `MessageView()` is unchanged.
    init(
        viewModel: ViewModel? = nil,
        presentedCardMessage: ChatMessage? = nil,
        dockResolveProgress: Double? = nil
    ) {
        _viewModel = State(initialValue: viewModel ?? ViewModel())
        _presentedCardMessage = State(initialValue: presentedCardMessage)
        storyDockResolveProgress = dockResolveProgress
    }

    var body: some View {
        NavigationStack {
            chatScrollView
                .safeAreaInset(edge: .bottom) {
                    composer
                }
                // The page background, full-bleed: the scroll view's own
                // background stops at its safe-area-adjusted bottom edge
                // (the composer's inset plus the field glass's reservation),
                // and without this the window shows through as a white band
                // above the composer.
                .background(Color(.systemGroupedBackground).ignoresSafeArea())
                .navigationBarTitleDisplayMode(.inline)
            .toolbar(.hidden, for: .navigationBar)
            .sheet(item: $presentedCardMessage) { message in
                MessageCardDetailSheet(message: message)
            }
            .coordinateSpace(name: messageRootCoordinateSpace)
            .onPreferenceChange(MessageFramePreferenceKey.self) { frames in
                messageFrames = frames.mapValues(\.root)
                messageScrollFrames = frames.mapValues(\.scroll)
            }
            .onPreferenceChange(ChatViewportFramePreferenceKey.self) { frame in
                chatViewportFrame = frame
            }
        }
        .task {
            await viewModel.connect()
            // Story mode only (#2965): applied after `connect()` loads the
            // fixture messages, so the frozen resolve snapshot has a real
            // message to point at — see the matching note in `.onAppear`.
            if storyDockResolveProgress != nil {
                viewModel.applyStoryDockResolveSnapshot()
            }
        }
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .background {
                isComposerFocused = false
            }
            viewModel.handleScenePhaseChange(newPhase)
        }
        .onDisappear {
            keyboardSettledScrollTask?.cancel()
        }
    }

    private var chatScrollView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 12) {
                    olderMessagesLoader(proxy: proxy)

                    topMessageLoader(proxy: proxy)

                    if viewModel.displayedMessages.isEmpty {
                        EmptyMessageFilterView(filter: .all)
                    } else {
                        messageGrid(layout: .message) { message in
                            scrollToMessageTopIfNeeded(message.id, with: proxy)
                        }
                    }

                    if viewModel.dockPhase != .hidden, viewModel.shouldShowLiveDock {
                        dockSlot
                    }

                    bottomSpacer
                }
                .padding(.horizontal, MessageLayout.message.horizontalPadding)
                .padding(.top, messageListTopPadding)
                .padding(.bottom, Tokens.composerFadeBleed)
            }
            .background {
                GeometryReader { geometry in
                    Color(.systemGroupedBackground)
                        .preference(
                            key: ChatViewportFramePreferenceKey.self,
                            value: CGRect(origin: .zero, size: geometry.size)
                        )
                }
            }
            .coordinateSpace(name: messageScrollCoordinateSpace)
            .scrollDismissesKeyboard(.interactively)
            // The dissolve ramp: content scrolling out at the composer's
            // edge fades into the page background over the same gap the
            // last message keeps below it.
            .overlay(alignment: .bottom) {
                if viewModel.hasHiddenNewMessages {
                    showNewMessagesPill
                }
            }
            .onAppear {
                scheduleScrollToBottom(with: proxy, animated: false)
                // `onChange(of:)` only fires on a later change, never for
                // the value a view already holds on its first appearance —
                // a run already in `agentProgressRuns` before the view
                // exists (restoring into an active run; every story fixture)
                // needs this explicit sync, or the dock never shows at all.
                // The resolving-snapshot story is the one exception: it
                // also needs `displayedMessages`, which `connect()` loads
                // asynchronously — applying it here would race that load
                // and resolve to no message at all, so it waits for the
                // `.task` below instead.
                if storyDockResolveProgress == nil {
                    viewModel.syncDock()
                }
            }
            .onChange(of: viewModel.foregroundProgressVersion) { _, _ in
                viewModel.syncDock()
            }
            .onChange(of: viewModel.displayedMessages.map(\.id)) { previousIDs, currentIDs in
                let previousPendingMessageID = pendingIncomingAutoScrollMessageID
                prepareArrivalAnimations(previousIDs: previousIDs, currentIDs: currentIDs)
                if pendingIncomingAutoScrollMessageID != previousPendingMessageID && viewModel.isAtBottom {
                    scheduleScrollToBottom(with: proxy, animated: false)
                }
            }
            .onChange(of: viewModel.scrollToBottomRequest) { _, _ in
                scheduleScrollToBottom(
                    with: proxy,
                    animated: false,
                    shouldSettleLayout: viewModel.shouldSettleScrollToBottom
                )
            }
            .onChange(of: viewModel.search.currentMatchID) { _, matchID in
                scrollToSearchMatch(matchID, with: proxy)
            }
        }
    }

    /// Find-in-chat (#3045): pages further back if the match isn't loaded
    /// yet, then centres it — the same centring the mock uses for the
    /// current match, `UnitPoint(x: 0.5, y: 0.45)`.
    private func scrollToSearchMatch(_ matchID: Int?, with proxy: ScrollViewProxy) {
        guard let matchID else { return }
        Task {
            await viewModel.loadUntilSearchMatchIsPresent(matchID)
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(matchID, anchor: UnitPoint(x: 0.5, y: 0.45))
            }
        }
    }

    /// The one flow slot the dock and its resolve occupy — never two
    /// places at once. `.working` shows the dock; `.resolving` shows the
    /// message it's becoming, IN THE SAME SLOT, both tagged with the same
    /// `matchedGeometryEffect` id so the dock's frame interpolates into the
    /// bubble's frame rather than the bubble appearing somewhere else
    /// while the dock separately fades at the bottom. That interpolation
    /// only runs inside the explicit `withAnimation` at the one call site
    /// that starts a resolve (`prepareArrivalAnimations`) — not a blanket
    /// `.animation(value:)` here, so the later, purely-bookkeeping
    /// `.resolving → .hidden` cleanup (the grid has already resumed
    /// showing this message by then) never re-animates anything.
    ///
    /// `matchedGeometryEffect` interpolates geometry only, never a shared
    /// glass identity — the fix for the crash a shared `glassEffectID`
    /// caused when two views claimed one identity simultaneously.
    @ViewBuilder
    private var dockSlot: some View {
        switch viewModel.dockPhase {
        case .hidden:
            EmptyView()
        case .working:
            if let dockRun = viewModel.dockRun {
                LiveDock(
                    items: viewModel.dockStreamItems,
                    step: dockRun.steps.last?.number ?? 1,
                    startedAt: dockRun.startedAt,
                    queued: viewModel.queuedMessageIDs.count,
                    permission: viewModel.dockPermission
                ) {
                    ChatTabAvatar()
                }
                .matchedGeometryEffect(id: dockGeometryID, in: dockNamespace)
                .transition(.opacity)
            }
        case .resolving:
            resolvingSlotContent
        }
    }

    /// Story mode only (#2965): a live `matchedGeometryEffect` interpolation
    /// can't be paused at a reproducible instant for a screenshot, so the
    /// frozen story instead blends the dock and the reply at fixed,
    /// complementary opacities, both in this same slot (never one above
    /// the other) — `dockResolveProgress` is the only path into this branch.
    @ViewBuilder
    private var resolvingSlotContent: some View {
        if let progress = storyDockResolveProgress, let dockRun = viewModel.dockRun, let message = resolvingMessage {
            ZStack(alignment: .topLeading) {
                LiveDock(
                    items: dockRun.liveDockItems,
                    step: dockRun.steps.last?.number ?? 1,
                    startedAt: dockRun.startedAt,
                    queued: viewModel.queuedMessageIDs.count
                ) {
                    ChatTabAvatar()
                }
                .opacity(1 - progress)
                messageGridCell(message, layout: .message, onTap: nil)
                    .opacity(progress)
            }
        } else if let message = resolvingMessage {
            messageGridCell(message, layout: .message, onTap: nil)
                .matchedGeometryEffect(id: dockGeometryID, in: dockNamespace)
                .transition(.opacity)
        }
    }

    /// The message the dock is resolving into, looked up from
    /// `displayedMessages` by id — `messageGridRows` excludes this same
    /// message from the normal grid while it's showing here instead.
    private var resolvingMessage: ChatMessage? {
        guard let id = viewModel.resolvingMessageID else { return nil }
        return viewModel.displayedMessages.first(where: { $0.id == id })
    }

    private func prepareArrivalAnimations(previousIDs: [Int], currentIDs: [Int]) {
        guard !previousIDs.isEmpty, currentIDs.count > previousIDs.count else { return }
        let appendedIDs = currentIDs.dropFirst(previousIDs.count)
        guard !appendedIDs.isEmpty else { return }

        let appendedIDSet = Set(appendedIDs)
        let appendedMessages = viewModel.displayedMessages.filter { appendedIDSet.contains($0.id) }
        let incomingIDs = appendedMessages.filter { !$0.isOutgoing }.map(\.id)
        guard !incomingIDs.isEmpty else { return }

        arrivalAnimatedMessageIDs.formUnion(incomingIDs)
        pendingIncomingAutoScrollMessageID = incomingIDs.last
        withAnimation(.easeInOut(duration: Tokens.dockMorphDuration)) {
            viewModel.beginResolvingIfMatched(appendedMessages: appendedMessages)
        }
    }

    /// The old toolbar's "show new messages" button, kept reachable as a
    /// floating pill over the chat canvas (#3037) — it is a filter action,
    /// not navigation.
    private var showNewMessagesPill: some View {
        Button {
            Task {
                await viewModel.clearFiltersAndShowNewMessages()
            }
        } label: {
            Label("Show new messages", systemImage: "message.badge")
                .font(.subheadline.weight(.semibold))
                .labelStyle(.titleAndIcon)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .glassEffect(.regular, in: .capsule)
        }
        .buttonStyle(.plain)
        .padding(.bottom, Tokens.composerFadeBleed)
        .accessibilityLabel("Show new messages")
    }

    /// The composer as a bottom safe-area inset: the scroll content clamps
    /// above it, the field's glass never has chat content behind it, and it
    /// rides the keyboard. The page background behind it is opaque, so the
    /// glass reads clean; the dissolve into it happens in the scroll view's
    /// own bottom ramp.
    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let replyMessage = viewModel.replyMessage {
                replyPreview(for: replyMessage)
            }

            Composer(
                text: $viewModel.draftMessage,
                placeholder: viewModel.composerPresentation.placeholder,
                tint: viewModel.composerPresentation.tint,
                pendingImage: viewModel.pendingImage,
                canSend: viewModel.client.canSend,
                isFocused: $isComposerFocused,
                onSend: { viewModel.sendDraft() },
                onImagePicked: { viewModel.pendingImage = $0 },
                onRemovePendingImage: { viewModel.pendingImage = nil }
            )
            .animation(.easeInOut(duration: 0.2), value: viewModel.client.conversationState)
        }
        .background(Color(.systemGroupedBackground))
    }

    private func replyPreview(for message: ChatMessage) -> some View {
        HStack(alignment: .center, spacing: 10) {
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .fill(Color.accentColor)
                .frame(width: 3, height: 38)

            VStack(alignment: .leading, spacing: 1) {
                Text(message.isOutgoing ? "You" : (message.sourceHint ?? "Penny"))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                Text(viewModel.replySummary(for: message))
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Button {
                viewModel.cancelReply()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .accessibilityLabel("Cancel reply")
        }
        .frame(height: 52)
        .padding(.horizontal, 12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Replying to \(viewModel.replySummary(for: message))")
    }

}

private extension MessageView {
    var bottomAnchorID: String { "message-list-bottom" }

    var messageScrollCoordinateSpace: String { "message-scroll-coordinate-space" }

    var messageRootCoordinateSpace: String { "message-root-coordinate-space" }

    var bottomSpacer: some View {
        Color.clear
            .frame(height: 1)
            .id(bottomAnchorID)
            .onAppear {
                viewModel.updateBottomVisibility(true)
            }
            .onDisappear {
                viewModel.updateBottomVisibility(false)
            }
    }

    @ViewBuilder
    func messageGrid(
        layout: MessageLayout,
        onMessageTap: ((ChatMessage) -> Void)? = nil
    ) -> some View {
        Grid(horizontalSpacing: layout.itemSpacing, verticalSpacing: layout.itemSpacing) {
            ForEach(messageGridRows(for: layout)) { row in
                GridRow {
                    ForEach(row.messages) { message in
                        messageGridCell(message, layout: layout, onTap: onMessageTap)
                    }

                    ForEach(row.messages.count..<layout.columnCount, id: \.self) { _ in
                        Color.clear
                            .frame(maxWidth: .infinity)
                            .accessibilityHidden(true)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity)
    }

    func messageGridCell(
        _ message: ChatMessage,
        layout: MessageLayout,
        onTap: ((ChatMessage) -> Void)?
    ) -> some View {
        VStack(spacing: 0) {
            Color.clear
                .frame(height: 0)
                .id(MessageTopAnchorID(messageID: message.id))
                .accessibilityHidden(true)

            ChatMessageView(
                message: message,
                layout: layout,
                isQueued: viewModel.isMessageQueued(message.id),
                searchQuery: viewModel.search.highlightQuery,
                isCurrentSearchMatch: message.id == viewModel.search.currentMatchID
            )
            .id(message.id)
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .background {
                GeometryReader { geometry in
                    Color.clear.preference(
                        key: MessageFramePreferenceKey.self,
                        value: [message.id: MessageFrameGeometry(
                            root: geometry.frame(in: .named(messageRootCoordinateSpace)),
                            scroll: geometry.frame(in: .named(messageScrollCoordinateSpace))
                        )]
                    )
                }
            }
            .contentShape(Rectangle())
            .onTapGesture {
                if layout == .message {
                    onTap?(message)
                    return
                }
                presentedCardMessage = message
            }
            .contextMenu {
                messageContextMenu(for: message)
            }
        }
        .modifier(IncomingMessageArrivalModifier(isEnabled: arrivalAnimatedMessageIDs.contains(message.id)) {
            arrivalAnimatedMessageIDs.remove(message.id)
        })
    }

    func messageGridRows(for layout: MessageLayout) -> [MessageGridRow] {
        let columnCount = layout.columnCount
        let messages = gridMessages
        return stride(from: 0, to: messages.count, by: columnCount).map { startIndex in
            let endIndex = min(startIndex + columnCount, messages.count)
            return MessageGridRow(messages: Array(messages[startIndex..<endIndex]))
        }
    }

    /// The messages the normal grid draws — every one of `displayedMessages`
    /// except, while the dock is resolving into it, the one message its
    /// slot is showing instead (#3042): drawing it in both places at once
    /// would duplicate it. Only excluded when the dock is actually visible
    /// (`shouldShowLiveDock`) — on a filter where the dock never shows, the
    /// message belongs in the grid as normal, not nowhere.
    var gridMessages: [ChatMessage] {
        guard viewModel.dockPhase == .resolving, viewModel.shouldShowLiveDock,
              let resolvingID = viewModel.resolvingMessageID else {
            return viewModel.displayedMessages
        }
        return viewModel.displayedMessages.filter { $0.id != resolvingID }
    }

    @ViewBuilder
    func olderMessagesLoader(proxy: ScrollViewProxy) -> some View {
        if viewModel.isLoadingOlderMessages {
            ProgressView()
                .controlSize(.small)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
        }
    }

    func topMessageLoader(proxy: ScrollViewProxy) -> some View {
        Color.clear
            .frame(height: 0)
            .background {
                GeometryReader { geometry in
                    Color.clear
                        .preference(
                            key: TopMessageLoaderPreferenceKey.self,
                            value: geometry.frame(in: .named(messageScrollCoordinateSpace)).minY
                        )
                }
            }
            .onPreferenceChange(TopMessageLoaderPreferenceKey.self) { minY in
                guard minY >= 0 else { return }
                loadOlderMessages(with: proxy)
            }
    }

    func scheduleScrollToBottom(
        with proxy: ScrollViewProxy,
        animated: Bool = true,
        shouldSettleLayout: Bool = true
    ) {
        let delays: [TimeInterval] = shouldSettleLayout ? [0.05, 0.16, 0.35] : [0.05]

        for (index, delay) in delays.enumerated() {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                scrollToBottom(
                    with: proxy,
                    animated: animated && index == 0,
                    enableOlderPaging: index == delays.count - 1
                )
            }
        }
    }

    func scrollToBottom(
        with proxy: ScrollViewProxy,
        animated: Bool = true,
        enableOlderPaging: Bool = true
    ) {
        if let pendingIncomingAutoScrollMessageID {
            guard let destination = incomingMessageAutoScrollDestination(for: pendingIncomingAutoScrollMessageID) else {
                finishBottomScrollIfNeeded(enableOlderPaging: enableOlderPaging)
                return
            }

            switch destination {
            case .messageTop(let target):
                scrollToMessageTop(target, with: proxy, animated: animated)
            case .bottomAnchor:
                scrollToBottomAnchor(with: proxy, animated: animated)
            }

            if enableOlderPaging {
                self.pendingIncomingAutoScrollMessageID = nil
            }
            finishBottomScrollIfNeeded(enableOlderPaging: enableOlderPaging)
            return
        }

        scrollToBottomAnchor(with: proxy, animated: animated)
        finishBottomScrollIfNeeded(enableOlderPaging: enableOlderPaging)
    }

    func scrollToBottomAnchor(with proxy: ScrollViewProxy, animated: Bool) {
        if animated {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(bottomAnchorID, anchor: .bottom)
            }
        } else {
            proxy.scrollTo(bottomAnchorID, anchor: .bottom)
        }
    }

    func scrollToMessageTopIfNeeded(_ id: Int, with proxy: ScrollViewProxy) {
        guard lastTopScrolledMessageID != id, shouldScrollToMessage(id) else { return }

        lastTopScrolledMessageID = id
        scrollToMessageTop(
            OversizedIncomingMessageScrollTarget(id: id),
            with: proxy,
            animated: true
        )
    }

    func scrollToMessageTop(_ target: OversizedIncomingMessageScrollTarget, with proxy: ScrollViewProxy, animated: Bool) {
        let anchor = messageTopScrollAnchor

        if animated {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(target.topAnchorID, anchor: anchor)
            }
        } else {
            proxy.scrollTo(target.topAnchorID, anchor: anchor)
        }
    }

    var messageTopScrollAnchor: UnitPoint {
        guard chatViewportFrame.height > 0 && chatViewportTopOcclusion > 0 else { return .top }
        return UnitPoint(x: 0.5, y: min(1, chatViewportTopOcclusion / chatViewportFrame.height))
    }

    var visibleChatViewportFrame: CGRect {
        guard chatViewportFrame.height > 0 else { return .zero }

        var frame = chatViewportFrame
        frame.origin.y += chatViewportTopOcclusion
        frame.size.height = max(0, frame.height - chatViewportTopOcclusion)
        return frame
    }

    var chatViewportTopOcclusion: CGFloat {
        0
    }

    var messageListTopPadding: CGFloat {
        12 + shellContentTopClearance
    }

    func shouldScrollToMessage(_ id: Int) -> Bool {
        guard let frame = messageScrollFrames[id], visibleChatViewportFrame.height > 0 else { return false }
        return frame.minY < visibleChatViewportFrame.minY || frame.maxY > visibleChatViewportFrame.maxY
    }

    func incomingMessageAutoScrollDestination(for id: Int) -> IncomingMessageAutoScrollDestination? {
        guard let message = viewModel.displayedMessages.last(where: { $0.id == id }) else { return nil }
        guard !message.isOutgoing else { return .bottomAnchor }
        guard let frame = messageFrames[id], visibleChatViewportFrame.height > 0 else { return nil }

        let availableHeight = max(44, visibleChatViewportFrame.height - 12)
        guard frame.height > availableHeight else { return .bottomAnchor }

        return .messageTop(OversizedIncomingMessageScrollTarget(id: id))
    }

    func finishBottomScrollIfNeeded(enableOlderPaging: Bool) {
        if enableOlderPaging && !viewModel.displayedMessages.isEmpty {
            viewModel.enableOlderPaging()
        }
    }

    func loadOlderMessages(with proxy: ScrollViewProxy) {
        guard let anchorID = viewModel.reserveOlderMessageLoad() else { return }
        Task {
            guard await viewModel.loadReservedOlderMessages() else { return }
            try? await Task.sleep(for: .milliseconds(16))
            var transaction = Transaction()
            transaction.disablesAnimations = true
            withTransaction(transaction) {
                proxy.scrollTo(anchorID, anchor: .top)
            }
            viewModel.finishOlderMessageScrollRestoration()
        }
    }

    /// The system long-press menu (#3038): Reply, Copy, and View steps (when
    /// the message carries a run). Replaces the custom overlay menu; its
    /// "Agent activity" item retired with the typing row and the activity
    /// sheet (#3042) — the live dock replaces what it showed.
    @ViewBuilder
    func messageContextMenu(for message: ChatMessage) -> some View {
        Button {
            withTransaction(Transaction(animation: .easeOut(duration: 0.12))) {
                viewModel.startReply(to: message)
            }
            Task { @MainActor in
                await Task.yield()
                isComposerFocused = true
            }
        } label: {
            Label("Reply", systemImage: "arrowshape.turn.up.left")
        }

        Button {
            UIPasteboard.general.string = message.content
        } label: {
            Label("Copy", systemImage: "doc.on.doc")
        }

        if let runID = message.runID {
            Button {
                isComposerFocused = false
                shellOpenPrompts(runID)
            } label: {
                Label("View steps", systemImage: "text.bubble")
            }
        }
    }
}

private enum IncomingMessageAutoScrollDestination {
    case messageTop(OversizedIncomingMessageScrollTarget)
    case bottomAnchor
}

private struct OversizedIncomingMessageScrollTarget {
    let id: Int

    var topAnchorID: MessageTopAnchorID {
        MessageTopAnchorID(messageID: id)
    }
}

private struct MessageTopAnchorID: Hashable {
    let messageID: Int
}

private struct MessageFrameGeometry: Equatable {
    let root: CGRect
    let scroll: CGRect
}

private struct MessageFramePreferenceKey: PreferenceKey {
    static let defaultValue: [Int: MessageFrameGeometry] = [:]

    static func reduce(value: inout [Int: MessageFrameGeometry], nextValue: () -> [Int: MessageFrameGeometry]) {
        value.merge(nextValue(), uniquingKeysWith: { _, newValue in newValue })
    }
}

private struct MessageGridRow: Identifiable {
    let messages: [ChatMessage]

    var id: Int {
        messages.first?.id ?? 0
    }
}

private struct ChatViewportFramePreferenceKey: PreferenceKey {
    static let defaultValue: CGRect = .zero

    static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
        let next = nextValue()
        value = next.height > 0 ? next : value
    }
}

private struct TopMessageLoaderPreferenceKey: PreferenceKey {
    static let defaultValue: CGFloat = -.infinity

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

struct MessageCardDetailSheet: View {
    let message: ChatMessage

    private var title: String {
        guard let sourceHint = message.sourceHint, !sourceHint.isEmpty else {
            return message.isOutgoing ? "You" : "Message"
        }
        return sourceHint
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                ChatMessageView(message: message, layout: .message)
                    .padding(.horizontal, MessageView.MessageLayout.message.horizontalPadding)
                    .padding(.vertical, 16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

private struct EmptyMessageFilterView: View {
    let filter: MessageView.MessageFilter

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: filter.systemImage)
                .font(.title2)
                .foregroundStyle(.secondary)

            Text(emptyText)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 48)
        .accessibilityElement(children: .combine)
    }

    private var emptyText: String {
        switch filter {
        case .all:
            return "No messages yet"
        case .penny:
            return "No Penny messages"
        case .chat:
            return "No chat messages"
        case .notifier:
            return "No notifier messages"
        case .collector:
            return "No collector messages"
        }
    }
}

private struct IncomingMessageArrivalModifier: ViewModifier {
    let isEnabled: Bool
    let onAnimationFinished: () -> Void

    @State private var hasArrived = false

    func body(content: Content) -> some View {
        content
            .opacity(isEnabled && !hasArrived ? 0 : 1)
            .offset(x: isEnabled && !hasArrived ? -28 : 0)
            .onAppear(perform: startAnimationIfNeeded)
            .onChange(of: isEnabled) { _, _ in
                startAnimationIfNeeded()
            }
    }

    private func startAnimationIfNeeded() {
        guard isEnabled, !hasArrived else { return }
        withAnimation(.spring(response: 0.34, dampingFraction: 0.86)) {
            hasArrived = true
        }
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(420))
            guard !Task.isCancelled else { return }
            onAnimationFinished()
        }
    }
}

#Preview {
    MessageView()
}
