import SwiftUI
import UIKit
import PennyServices

struct MessageView: View {
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.shellContentTopClearance) private var shellContentTopClearance
    @Environment(\.shellOpenPrompts) private var shellOpenPrompts
    @State private var viewModel: ViewModel
    @State private var isShowingActivity: Bool
    @State private var presentedCardMessage: ChatMessage?
    @State private var isShowingSearchStory: Bool
    @State private var messageFrames: [Int: CGRect] = [:]
    @State private var messageScrollFrames: [Int: CGRect] = [:]
    @State private var keyboardSettledScrollTask: Task<Void, Never>?
    @State private var typingIndicatorStatus = "Penny is typing"
    @State private var isTypingIndicatorVisible = false
    @State private var isTypingIndicatorFadingOut = false
    @State private var typingIndicatorFadeTask: Task<Void, Never>?
    @State private var arrivalAnimatedMessageIDs: Set<Int> = []
    @State private var pendingIncomingAutoScrollMessageID: Int?
    @State private var lastTopScrolledMessageID: Int?
    @State private var chatViewportFrame: CGRect = .zero
    @FocusState private var isComposerFocused: Bool

    /// `isShowingActivity`, `presentedCardMessage`, and `isShowingSearchStory`
    /// only exist so story mode (#2965) can open the activity sheet, the card
    /// detail sheet, or the search screen already presented over the chat
    /// story. They default to today's behaviour, so `MessageView()` is
    /// unchanged, and `isShowingSearchStory` is a seam separate from the real
    /// `NavigationLink` search entry point below — it never changes that path.
    init(
        viewModel: ViewModel? = nil,
        isShowingActivity: Bool = false,
        presentedCardMessage: ChatMessage? = nil,
        isShowingSearchStory: Bool = false
    ) {
        _viewModel = State(initialValue: viewModel ?? ViewModel())
        _isShowingActivity = State(initialValue: isShowingActivity)
        _presentedCardMessage = State(initialValue: presentedCardMessage)
        _isShowingSearchStory = State(initialValue: isShowingSearchStory)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                chatScrollView
            }
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
            .safeAreaInset(edge: .bottom) {
                composer
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar(.hidden, for: .navigationBar)
            .sheet(isPresented: $isShowingActivity) {
                AgentActivityView(client: viewModel.client)
            }
            .sheet(item: $presentedCardMessage) { message in
                MessageCardDetailSheet(message: message)
            }
            .navigationDestination(isPresented: $isShowingSearchStory) {
                MessageSearchView(client: viewModel.client) { message in
                    withTransaction(Transaction(animation: .easeOut(duration: 0.12))) {
                        viewModel.startReply(to: message)
                    }
                }
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
        }
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .background {
                isComposerFocused = false
            }
            viewModel.handleScenePhaseChange(newPhase)
        }
        .onDisappear {
            keyboardSettledScrollTask?.cancel()
            typingIndicatorFadeTask?.cancel()
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

                    if isTypingIndicatorVisible && viewModel.shouldShowTypingIndicator {
                        TypingRow(status: typingIndicatorStatus, isFadingOut: isTypingIndicatorFadingOut)
                    }

                    bottomSpacer
                }
                .padding(.horizontal, MessageLayout.message.horizontalPadding)
                .padding(.top, messageListTopPadding)
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
            .overlay(alignment: .bottom) {
                if viewModel.hasHiddenNewMessages {
                    showNewMessagesPill
                }
            }
            .onAppear {
                updateTypingIndicator(isTyping: viewModel.client.isTyping)
                scheduleScrollToBottom(with: proxy, animated: false)
            }
            .onChange(of: viewModel.client.isTyping) { _, isTyping in
                updateTypingIndicator(isTyping: isTyping)
                if viewModel.isAtBottom {
                    scheduleScrollToBottom(with: proxy, shouldSettleLayout: false)
                }
            }
            .onChange(of: viewModel.client.foregroundProgress?.currentStatus) { _, status in
                updateTypingIndicatorStatus(status)
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
        }
    }

    private func updateTypingIndicator(isTyping: Bool) {
        typingIndicatorFadeTask?.cancel()

        if isTyping {
            typingIndicatorStatus = currentTypingStatus
            isTypingIndicatorVisible = true
            withAnimation(.easeOut(duration: 0.16)) {
                isTypingIndicatorFadingOut = false
            }
            return
        }

        guard isTypingIndicatorVisible else { return }

        withAnimation(.easeOut(duration: 0.35)) {
            isTypingIndicatorFadingOut = true
        }

        typingIndicatorFadeTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled else { return }
            isTypingIndicatorVisible = false
            isTypingIndicatorFadingOut = false
        }
    }

    private func updateTypingIndicatorStatus(_ status: String?) {
        guard viewModel.client.isTyping else { return }
        typingIndicatorStatus = status ?? "Penny is typing"
    }

    private func prepareArrivalAnimations(previousIDs: [Int], currentIDs: [Int]) {
        guard !previousIDs.isEmpty, currentIDs.count > previousIDs.count else { return }
        let appendedIDs = currentIDs.dropFirst(previousIDs.count)
        guard !appendedIDs.isEmpty else { return }

        let appendedIDSet = Set(appendedIDs)
        let incomingIDs = viewModel.displayedMessages
            .filter { appendedIDSet.contains($0.id) && !$0.isOutgoing }
            .map(\.id)
        guard !incomingIDs.isEmpty else { return }

        arrivalAnimatedMessageIDs.formUnion(incomingIDs)
        pendingIncomingAutoScrollMessageID = incomingIDs.last
    }

    private var currentTypingStatus: String {
        viewModel.client.foregroundProgress?.currentStatus ?? "Penny is typing"
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
        .padding(.bottom, 12)
        .accessibilityLabel("Show new messages")
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let replyMessage = viewModel.replyMessage {
                replyPreview(for: replyMessage)
            }

            HStack(alignment: .bottom, spacing: 8) {
                TextField(
                    viewModel.composerPresentation.placeholder,
                    text: $viewModel.draftMessage,
                    axis: .vertical
                )
                    .textFieldStyle(.plain)
                    .font(.body)
                    .lineLimit(1...5)
                    .submitLabel(.send)
                    .focused($isComposerFocused)
                    .onSubmit(viewModel.sendDraft)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 12)
                    .glassEffect(
                        .regular.tint(viewModel.composerPresentation.tint),
                        in: .capsule
                    )
                    .animation(
                        .easeInOut(duration: 0.2),
                        value: viewModel.client.conversationState
                    )

                Button(action: viewModel.sendDraft) {
                    Image(systemName: "paperplane.fill")
                        .font(.system(size: 16, weight: .semibold))
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.glassProminent)
                .buttonBorderShape(.circle)
                .disabled(viewModel.draftMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !viewModel.client.canSend)
                .accessibilityLabel("Send")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.clear)
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

            ChatMessageView(message: message, layout: layout)
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
        return stride(from: 0, to: viewModel.displayedMessages.count, by: columnCount).map { startIndex in
            let endIndex = min(startIndex + columnCount, viewModel.displayedMessages.count)
            return MessageGridRow(messages: Array(viewModel.displayedMessages[startIndex..<endIndex]))
        }
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

    /// The system long-press menu (#3038): Reply, Copy, View steps (when
    /// the message carries a run), and Agent activity — which stays
    /// reachable until the live dock (#3042) replaces it. Replaces the
    /// custom overlay menu.
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

        Button {
            isShowingActivity = true
        } label: {
            Label("Agent activity", systemImage: "waveform.path")
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

struct TypingRow: View {
    let status: String
    let isFadingOut: Bool

    var body: some View {
        HStack(alignment: .center, spacing: 6) {
            Text(status)
                .id(status)
                .font(.subheadline)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .transition(.opacity)
            TypingDots()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .frame(maxWidth: .infinity, alignment: .leading)
        .opacity(isFadingOut ? 0 : 1)
        .animation(.easeInOut(duration: 0.14), value: status)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(status)
    }
}

private struct TypingDots: View {
    private let dotCount = 3
    private let cycleDuration = 0.9
    private let dotDelay = 0.16
    private let jumpDuration = 0.36

    var body: some View {
        TimelineView(.animation) { timeline in
            let elapsedTime = timeline.date.timeIntervalSinceReferenceDate

            HStack(spacing: 3) {
                ForEach(0..<dotCount, id: \.self) { index in
                    let progress = dotProgress(elapsedTime: elapsedTime, index: index)

                    Circle()
                        .fill(Color.secondary)
                        .frame(width: 5, height: 5)
                        .offset(y: -4 * sin(progress * .pi))
                        .opacity(0.45 + (0.35 * sin(progress * .pi)))
                }
            }
        }
        .frame(width: 21, height: 12)
        .accessibilityHidden(true)
    }

    private func dotProgress(elapsedTime: TimeInterval, index: Int) -> Double {
        let delayedTime = elapsedTime - (Double(index) * dotDelay)
        let cycleTime = delayedTime.truncatingRemainder(dividingBy: cycleDuration)
        let normalizedCycleTime = cycleTime < 0 ? cycleTime + cycleDuration : cycleTime

        guard normalizedCycleTime < jumpDuration else {
            return 0
        }

        return normalizedCycleTime / jumpDuration
    }
}

private struct AgentActivityView: View {
    let client: PennyService

    var body: some View {
        NavigationStack {
            List {
                Section("Connection") {
                    LabeledContent("Status", value: client.statusText)

                    if let error = client.lastError {
                        Text(error)
                            .font(.subheadline)
                            .foregroundStyle(.red)

                        Button("Reconnect", systemImage: "arrow.clockwise") {
                            client.reconnect()
                        }
                    }
                }

                if let foreground = client.foregroundProgress {
                    Section("Current message") {
                        progressRun(foreground)
                    }
                }

                Section("Background work") {
                    if client.backgroundProgressRuns.isEmpty {
                        Text("Nothing is running")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(client.backgroundProgressRuns) { run in
                            progressRun(run)
                        }
                    }
                }
            }
            .navigationTitle("Agent activity")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    @ViewBuilder
    private func progressRun(_ run: AgentProgressRunItem) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(run.agent.capitalized)
                .font(.headline)
            ForEach(run.steps) { step in
                VStack(alignment: .leading, spacing: 4) {
                    Text("Step \(step.number)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    ForEach(step.tools) { tool in
                            Label(agentProgressToolLabel(tool), systemImage: "arrow.triangle.2.circlepath")
                            .font(.subheadline)
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }

}

extension AgentProgressRunItem {
    var currentStatus: String {
        guard let step = steps.last else { return "Penny is working" }
        let status = "Working • Step \(step.number)"
        if let tool = step.tools.last { return "\(status): \(agentProgressToolLabel(tool))" }
        return status
    }
}

private func agentProgressToolLabel(_ tool: AgentProgressToolItem) -> String {
    switch tool.name {
    case "browse":
        if case .array(let queries)? = tool.arguments["queries"],
           let query = queries.compactMap({ value in
               if case .string(let string) = value { return string }
               return nil
           }).first {
            return formattedURLHost(from: query).map { "Reading \($0)" } ?? "Searching \"\(query)\""
        }
        return "Looking up"
    case "search_emails": return "Searching emails"
    case "read_emails": return "Reading emails"
    case "list_emails": return "Listing emails"
    case "list_folders": return "Listing email folders"
    case "draft_email": return "Drafting an email"
    case "generate_image": return "Generating an image"
    default:
        if case .string(let detail)? = tool.arguments["detail"], let host = formattedURLHost(from: detail) {
            return "Using \(tool.name) · \(host)"
        }
        return "Using \(tool.name)"
    }
}

private func formattedURLHost(from string: String) -> String? {
    guard let components = URLComponents(string: string), let host = components.host else { return nil }
    return host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
}

#Preview {
    MessageView()
}

#Preview("Progress sizes") {
    VStack(spacing: 12) {
        TypingRow(status: "Working • Step 1", isFadingOut: false)
        TypingRow(status: "Working • Step 1: Reading example.com", isFadingOut: false)
        TypingRow(status: "Working • Step 2: Searching emails", isFadingOut: false)
        TypingRow(status: "Working • Step 3: Searching \"a longer query that wraps onto multiple lines\"", isFadingOut: false)
    }
    .padding()
    .background(Color(.systemGroupedBackground))
}
