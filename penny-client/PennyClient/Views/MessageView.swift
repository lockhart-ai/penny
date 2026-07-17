import SwiftUI
import UIKit

struct MessageView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var viewModel = ViewModel()
    @State private var isShowingActivity = false
    @State private var presentedCardMessage: ChatMessage?
    @State private var activeMessageContext: MessageActionContext?
    @State private var messageFrames: [Int: CGRect] = [:]
    @State private var messageScrollFrames: [Int: CGRect] = [:]
    @State private var messageActionProxyHeights: [Int: CGFloat] = [:]
    @State private var messageContextScale: CGFloat = 1
    @State private var keyboardSettledScrollTask: Task<Void, Never>?
    @State private var typingIndicatorStatus = "Penny is typing"
    @State private var isTypingIndicatorVisible = false
    @State private var isTypingIndicatorFadingOut = false
    @State private var typingIndicatorFadeTask: Task<Void, Never>?
    @State private var arrivalAnimatedMessageIDs: Set<Int> = []
    @State private var pendingIncomingAutoScrollMessageID: Int?
    @State private var lastTopScrolledMessageID: Int?
    @State private var chatViewportFrame: CGRect = .zero
    @State private var selectedPennyNavigation: PennyNavigationDestination?
    @FocusState private var isComposerFocused: Bool

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
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 8) {
                        pennyNavigationMenu

                        if viewModel.hasHiddenNewMessages {
                            hiddenNewMessagesButton
                        }
                    }
                }

                ToolbarItem(placement: .principal) {
                    titleBar
                }

                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 14) {
                        NavigationLink {
                            MessageSearchView(client: viewModel.client) { message in
                                withTransaction(Transaction(animation: .easeOut(duration: 0.12))) {
                                    viewModel.startReply(to: message)
                                }
                            }
                        } label: {
                            Image(systemName: "magnifyingglass")
                                .frame(width: 28, height: 28)
                                .contentShape(Circle())
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.primary)
                        .simultaneousGesture(TapGesture().onEnded {
                            isComposerFocused = false
                        })
                        .accessibilityLabel("Search messages")

                        Button {
                            viewModel.isShowingSettings = true
                        } label: {
                            Image(systemName: "gearshape")
                                .frame(width: 28, height: 28)
                                .contentShape(Circle())
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.primary)
                        .accessibilityLabel("Settings")

                    }
                }
            }
            .sheet(isPresented: $viewModel.isShowingSettings) {
                SettingsView(client: viewModel.client)
            }
            .sheet(isPresented: $isShowingActivity) {
                AgentActivityView(client: viewModel.client)
            }
            .navigationDestination(item: $selectedPennyNavigation) { destination in
                pennyNavigationDestination(destination)
            }
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
            .overlay {
                messageActionOverlay
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

    private var pennyNavigationMenu: some View {
        Menu {
            ForEach(PennyNavigationDestination.allCases) { destination in
                Button {
                    selectedPennyNavigation = destination
                } label: {
                    Label(destination.title, systemImage: destination.systemImage)
                }
            }
        } label: {
            Image(systemName: "memories")
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.primary)
        .accessibilityLabel("Penny navigation")
    }

    @ViewBuilder
    private func pennyNavigationDestination(_ destination: PennyNavigationDestination) -> some View {
        switch destination {
        case .insights:
            InsightsView(client: viewModel.client)
        case .memories:
            MemoryManagementView(client: viewModel.client)
        }
    }

    private var hiddenNewMessagesButton: some View {
        Button {
            Task {
                await viewModel.clearFiltersAndShowNewMessages()
            }
        } label: {
            Image(systemName: "message.badge")
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(Color.accentColor)
        .accessibilityLabel("Show new messages")
    }

    private var titleBar: some View {
        Button {
            isShowingActivity = true
        } label: {
            HStack(spacing: 8) {
                Image("penny")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 24, height: 24)

                Text("Penny")
                    .font(.headline)

                StatusIndicator(
                    color: viewModel.client.connectionColor,
                    statusText: viewModel.client.statusText,
                    isActive: hasAgentActivity
                )
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Penny activity")
        .accessibilityValue(viewModel.client.statusText)
    }

    private var hasAgentActivity: Bool {
        viewModel.client.isTyping || !viewModel.client.agentProgressRuns.isEmpty
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let replyMessage = viewModel.replyMessage {
                replyPreview(for: replyMessage)
            }

            HStack(alignment: .bottom, spacing: 8) {
                TextField("Message", text: $viewModel.draftMessage, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.body)
                    .lineLimit(1...5)
                    .submitLabel(.send)
                    .focused($isComposerFocused)
                    .onSubmit(viewModel.sendDraft)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 12)
                    .glassEffect(.regular, in: .capsule)

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

private enum PennyNavigationDestination: String, CaseIterable, Identifiable {
    case insights
    case memories

    var id: Self { self }

    var title: String {
        switch self {
        case .insights:
            return "Insights"
        case .memories:
            return "Memories"
        }
    }

    var systemImage: String {
        switch self {
        case .insights:
            return "chart.bar.doc.horizontal"
        case .memories:
            return "tray.full"
        }
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
                .opacity(activeMessageContext?.message.id == message.id ? 0 : 1)
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
                    if activeMessageContext?.message.id == message.id {
                        dismissMessageActions()
                        return
                    }
                    if layout == .message {
                        onTap?(message)
                        return
                    }
                    presentedCardMessage = message
                }
                .onLongPressGesture(minimumDuration: 0.35) {
                    presentMessageActions(for: message)
                }
        }
        .modifier(IncomingMessageArrivalModifier(isEnabled: arrivalAnimatedMessageIDs.contains(message.id)) {
            arrivalAnimatedMessageIDs.remove(message.id)
        })
        .animation(.spring(response: 0.24, dampingFraction: 0.78), value: activeMessageContext?.message.id)
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
        12
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

    @ViewBuilder
    var messageActionOverlay: some View {
        if let context = activeMessageContext {
            GeometryReader { geometry in
                ZStack(alignment: .topLeading) {
                    Rectangle()
                        .fill(.regularMaterial)
                        .opacity(0.86)
                        .ignoresSafeArea()
                        .onTapGesture(perform: dismissMessageActions)

                    messageActionStack(for: context, in: geometry.size)
                }
                .onPreferenceChange(MessageActionProxyHeightPreferenceKey.self) { heights in
                    messageActionProxyHeights.merge(heights, uniquingKeysWith: { _, newValue in newValue })
                }
            }
            .transition(.opacity)
        }
    }

    func messageActionStack(for context: MessageActionContext, in containerSize: CGSize) -> some View {
        let stackWidth = actionProxyWidth(for: context, in: containerSize)
        let estimatedMenuHeight: CGFloat = 112
        let stackSpacing: CGFloat = 10
        let availableStackHeight = max(1, containerSize.height - 32)
        let maximumProxyHeight = max(44, availableStackHeight - estimatedMenuHeight - stackSpacing)
        let measuredProxyHeight = messageActionProxyHeights[context.id]
        let shouldScrollProxy = shouldScrollMessageActionProxy(
            for: context,
            measuredHeight: measuredProxyHeight,
            maxHeight: maximumProxyHeight
        )
        let fallbackProxyHeight = context.frame.height
        let proxyHeight = shouldScrollProxy ? maximumProxyHeight : min(maximumProxyHeight, max(1, measuredProxyHeight ?? fallbackProxyHeight))
        let stackHeight = proxyHeight + estimatedMenuHeight + stackSpacing
        let leading = actionProxyLeading(for: context, width: stackWidth, in: containerSize)
        let maximumTop = max(16, containerSize.height - stackHeight - 16)
        let top = min(max(16, context.frame.minY), maximumTop)
        let messageAlignment: Alignment = context.message.isOutgoing ? .trailing : .leading

        return VStack(alignment: context.message.isOutgoing ? .trailing : .leading, spacing: stackSpacing) {
            messageActionProxy(
                for: context,
                layout: MessageActionProxyLayout(
                    width: stackWidth,
                    height: proxyHeight,
                    measuredHeight: measuredProxyHeight,
                    maxHeight: maximumProxyHeight,
                    alignment: messageAlignment
                )
            )
            .scaleEffect(messageContextScale)

            messageActionMenu(for: context.message)
                .frame(width: 220)
        }
        .frame(width: stackWidth, alignment: messageAlignment)
        .background {
            messageActionProxyMeasurement(for: context, width: stackWidth, alignment: messageAlignment)
        }
        .offset(x: leading, y: top)
        .animation(.spring(response: 0.22, dampingFraction: 0.72), value: messageContextScale)
    }

    func messageActionProxyMeasurement(for context: MessageActionContext, width: CGFloat, alignment: Alignment) -> some View {
        ChatMessageView(message: context.message, layout: .message)
            .frame(width: width, alignment: alignment)
            .fixedSize(horizontal: false, vertical: true)
            .background {
                GeometryReader { geometry in
                    Color.clear.preference(
                        key: MessageActionProxyHeightPreferenceKey.self,
                        value: [context.id: geometry.size.height]
                    )
                }
            }
            .opacity(0)
            .allowsHitTesting(false)
    }

    func actionProxyWidth(for context: MessageActionContext, in containerSize: CGSize) -> CGFloat {
        let maximumWidth = max(1, containerSize.width - MessageLayout.message.horizontalPadding * 2)
        return min(max(1, context.frame.width), maximumWidth)
    }

    func actionProxyLeading(for context: MessageActionContext, width: CGFloat, in containerSize: CGSize) -> CGFloat {
        let maximumLeading = max(MessageLayout.message.horizontalPadding, containerSize.width - width - MessageLayout.message.horizontalPadding)
        return min(max(MessageLayout.message.horizontalPadding, context.frame.minX), maximumLeading)
    }

    func shouldScrollMessageActionProxy(
        for context: MessageActionContext,
        measuredHeight: CGFloat?,
        maxHeight: CGFloat
    ) -> Bool {
        if let measuredHeight {
            return measuredHeight > maxHeight
        }
        return context.frame.height > maxHeight
    }

    @ViewBuilder
    func messageActionProxy(for context: MessageActionContext, layout: MessageActionProxyLayout) -> some View {
        if shouldScrollMessageActionProxy(for: context, measuredHeight: layout.measuredHeight, maxHeight: layout.maxHeight) {
            ScrollView {
                ChatMessageView(message: context.message, layout: .message)
                    .frame(width: layout.width, alignment: layout.alignment)
            }
            .frame(width: layout.width)
            .frame(height: layout.height, alignment: .top)
            .scrollIndicators(.hidden)
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .contentShape(Rectangle())
            .onTapGesture(perform: dismissMessageActions)
        } else {
            ChatMessageView(message: context.message, layout: .message)
                .frame(width: layout.width, alignment: layout.alignment)
                .contentShape(Rectangle())
                .onTapGesture(perform: dismissMessageActions)
        }
    }

    func messageActionMenu(for message: ChatMessage) -> some View {
        VStack(spacing: 0) {
            Button {
                withTransaction(Transaction(animation: .easeOut(duration: 0.12))) {
                    viewModel.startReply(to: message)
                    dismissMessageActions()
                }
                Task { @MainActor in
                    await Task.yield()
                    isComposerFocused = true
                }
            } label: {
                menuButtonRow(title: "Reply", systemImage: "arrowshape.turn.up.left")
            }

            Divider()
                .padding(.leading, 48)

            Button {
                UIPasteboard.general.string = message.content
                dismissMessageActions()
            } label: {
                menuButtonRow(title: "Copy", systemImage: "doc.on.doc")
            }
        }
        .buttonStyle(.plain)
        .font(.body)
        .foregroundStyle(.primary)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color(.separator).opacity(0.28), lineWidth: 0.5)
        }
        .shadow(color: .black.opacity(0.18), radius: 18, y: 8)
    }

    func menuButtonRow(title: String, systemImage: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: systemImage)
                .frame(width: 22)

            Text(title)

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.clear)
        .contentShape(Rectangle())
    }

    func presentMessageActions(for message: ChatMessage) {
        guard let frame = messageFrames[message.id] else { return }
        activeMessageContext = MessageActionContext(message: message, frame: frame)
        messageContextScale = 1.06
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
            guard activeMessageContext?.message.id == message.id else { return }
            messageContextScale = 1
        }
    }

    func dismissMessageActions() {
        if let activeMessageContext {
            messageActionProxyHeights[activeMessageContext.id] = nil
        }
        activeMessageContext = nil
        messageContextScale = 1
    }
}

private struct MessageActionContext: Identifiable {
    let message: ChatMessage
    let frame: CGRect

    var id: Int {
        message.id
    }
}

private struct MessageActionProxyLayout {
    let width: CGFloat
    let height: CGFloat
    let measuredHeight: CGFloat?
    let maxHeight: CGFloat
    let alignment: Alignment
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

private struct MessageActionProxyHeightPreferenceKey: PreferenceKey {
    static let defaultValue: [Int: CGFloat] = [:]

    static func reduce(value: inout [Int: CGFloat], nextValue: () -> [Int: CGFloat]) {
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
                ChatMessageView(message: message, layout: .message, showsSourceHintInline: false, fillsMessageRowWidth: true)
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

private struct StatusIndicator: View {
    let color: Color
    let statusText: String
    let isActive: Bool

    var body: some View {
        TimelineView(.animation) { timeline in
            Circle()
                .fill(color)
                .frame(width: size(at: timeline.date), height: size(at: timeline.date))
                .animation(.easeInOut(duration: 0.18), value: isActive)
        }
        .frame(width: 13, height: 13)
        .accessibilityLabel(statusText)
    }

    private func size(at date: Date) -> CGFloat {
        guard isActive else { return 9 }
        let elapsedTime = date.timeIntervalSinceReferenceDate
        let progress = (sin(elapsedTime * 4.6) + 1) / 2
        return 9 + (progress * 3)
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

private struct TypingRow: View {
    let status: String
    let isFadingOut: Bool

    var body: some View {
        HStack(alignment: .center, spacing: 6) {
            Text(status)
                .id(status)
                .font(.subheadline)
                .multilineTextAlignment(.leading)
                .transition(.opacity)
            Spacer(minLength: 8)
            TypingDots()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .opacity(isFadingOut ? 0 : 1)
        .animation(.easeInOut(duration: 0.14), value: status)
        .accessibilityLabel("Penny is typing")
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
                    Text(step.maxSteps.map { "Step \(step.number) of \($0)" } ?? "Step \(step.number)")
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

private extension AgentProgressRunItem {
    var currentStatus: String {
        guard let step = steps.last else { return "Penny is working" }
        if let tool = step.tools.last { return agentProgressToolLabel(tool) }
        return step.maxSteps.map { "Working · step \(step.number) of \($0)" } ?? "Working · step \(step.number)"
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
