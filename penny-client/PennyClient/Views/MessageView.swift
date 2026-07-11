import SwiftUI
import UIKit

struct MessageView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var viewModel = ViewModel()
    @State private var isMessageLayoutSwitcherEnabled = Prefs.shared.isMessageLayoutSwitcherEnabled
    @State private var presentedCardMessage: ChatMessage?
    @State private var activeMessageContext: MessageActionContext?
    @State private var messageFrames: [Int: CGRect] = [:]
    @State private var messageActionProxyHeights: [Int: CGFloat] = [:]
    @State private var messageContextScale: CGFloat = 1
    @State private var shouldUseFastComposerScroll = false
    @State private var keyboardSettledScrollTask: Task<Void, Never>?
    @State private var selectedPennyNavigation: PennyNavigationDestination?
    @FocusState private var isComposerFocused: Bool

    private var effectiveMessageLayout: MessageLayout {
        isMessageLayoutSwitcherEnabled ? viewModel.selectedMessageLayout : .message
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if effectiveMessageLayout == .message {
                    chatScrollView
                } else {
                    cardScrollView(layout: effectiveMessageLayout)
                }
            }
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
            .overlay(alignment: .top) {
                if isMessageLayoutSwitcherEnabled {
                    messageLayoutSelector
                        .padding(.top, 10)
                }
            }
            .safeAreaInset(edge: .bottom) {
                composer
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 8) {
                        messageFilterMenu
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
                        Button {
                            viewModel.isShowingSearch = true
                        } label: {
                            Image(systemName: "magnifyingglass")
                                .frame(width: 28, height: 28)
                                .contentShape(Circle())
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.primary)
                        .accessibilityLabel("Search messages")

                        if viewModel.client.lastError != nil {
                            Button {
                                viewModel.isShowingConnectionError = true
                            } label: {
                                Image(systemName: "info.circle")
                                    .frame(width: 28, height: 28)
                                    .contentShape(Circle())
                            }
                            .buttonStyle(.borderless)
                            .foregroundStyle(.primary)
                            .accessibilityLabel("Connection error")
                        }

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
            .alert("Connection Error", isPresented: $viewModel.isShowingConnectionError, presenting: viewModel.client.lastError) { _ in
                Button("Reconnect") {
                    viewModel.reconnect()
                }
                Button("OK", role: .cancel) {}
            } message: { errorMessage in
                Text(errorMessage)
            }
            .sheet(isPresented: $viewModel.isShowingSettings) {
                SettingsView(client: viewModel.client)
            }
            .sheet(isPresented: $viewModel.isShowingSearch) {
                MessageSearchView(client: viewModel.client)
            }
            .navigationDestination(item: $selectedPennyNavigation) { destination in
                pennyNavigationDestination(destination)
            }
            .sheet(item: $presentedCardMessage) { message in
                MessageCardDetailSheet(message: message)
            }
            .coordinateSpace(name: messageRootCoordinateSpace)
            .onPreferenceChange(MessageFramePreferenceKey.self) { frames in
                messageFrames = frames
            }
            .overlay {
                messageActionOverlay
            }
            .onChange(of: viewModel.isShowingSettings) { _, isShowingSettings in
                guard !isShowingSettings else { return }
                refreshFeaturePreferences()
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
        .onChange(of: isComposerFocused) { _, isFocused in
            handleComposerFocusChanged(isFocused)
        }
        .onDisappear {
            keyboardSettledScrollTask?.cancel()
            viewModel.disconnect()
        }
    }

    private var chatScrollView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 12) {
                    olderMessagesLoader(proxy: proxy)

                    topMessageLoader(proxy: proxy)

                    if viewModel.displayedMessages.isEmpty {
                        EmptyMessageFilterView(filter: viewModel.selectedMessageFilter)
                    } else {
                        messageGrid(layout: .message)
                    }

                    if viewModel.client.isTyping && viewModel.shouldShowTypingIndicator {
                        TypingRow()
                    }

                    bottomSpacer
                }
                .padding(.horizontal, MessageLayout.message.horizontalPadding)
                .padding(.top, isMessageLayoutSwitcherEnabled ? 58 : 12)
            }
            .background(Color(.systemGroupedBackground))
            .coordinateSpace(name: messageScrollCoordinateSpace)
            .scrollDismissesKeyboard(.interactively)
            .onAppear {
                scheduleScrollToBottom(with: proxy, animated: false)
            }
            .onChange(of: viewModel.client.isTyping) { _, _ in
                if viewModel.isAtBottom {
                    scheduleScrollToBottom(with: proxy, shouldSettleLayout: false)
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

    private func cardScrollView(layout: MessageLayout) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 12) {
                    olderMessagesLoader(proxy: proxy)

                    topMessageLoader(proxy: proxy)

                    if viewModel.displayedMessages.isEmpty {
                        EmptyMessageFilterView(filter: viewModel.selectedMessageFilter)
                    } else {
                        messageGrid(layout: layout)
                    }

                    bottomSpacer
                }
                .padding(.horizontal, layout.horizontalPadding)
                .padding(.top, isMessageLayoutSwitcherEnabled ? 58 : 12)
            }
            .background(Color(.systemGroupedBackground))
            .coordinateSpace(name: messageScrollCoordinateSpace)
            .scrollDismissesKeyboard(.interactively)
            .onAppear {
                scheduleScrollToBottom(with: proxy, animated: false)
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
        case .schedules:
            SchedulesView(client: viewModel.client)
        case .insights:
            InsightsView(client: viewModel.client)
        case .memories:
            MemoryManagementView(client: viewModel.client)
        }
    }

    private var messageFilterMenu: some View {
        Menu {
            Picker("Filter Messages", selection: $viewModel.selectedMessageFilter) {
                ForEach(MessageFilter.allCases) { filter in
                    Label(filter.title, systemImage: filter.systemImage)
                        .tag(filter)
                }
            }
        } label: {
            Image(systemName: viewModel.selectedMessageFilter == .all ? "line.3.horizontal.decrease.circle" : "line.3.horizontal.decrease.circle.fill")
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.primary)
        .accessibilityLabel("Filter messages")
        .accessibilityValue(viewModel.selectedMessageFilter.title)
    }

    private var hiddenNewMessagesButton: some View {
        Button {
            Task {
                await viewModel.clearFiltersAndShowNewMessages()
                viewModel.selectedMessageLayout = .message
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

    private var messageLayoutSelector: some View {
        HStack(spacing: 4) {
            ForEach(MessageLayout.allCases) { layout in
                Button {
                    changeMessageLayout(to: layout)
                } label: {
                    Image(systemName: viewModel.selectedMessageLayout == layout ? "\(layout.rawValue).circle.fill" : layout.systemImage)
                        .font(.system(size: 17, weight: .semibold))
                        .frame(width: 36, height: 32)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(viewModel.selectedMessageLayout == layout ? Color.accentColor : .secondary)
                .accessibilityLabel("\(layout.title) layout")
            }
        }
        .padding(4)
        .glassEffect(.regular, in: .capsule)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Message layout")
    }

    private var titleBar: some View {
        HStack(spacing: 8) {
            Image("penny")
                .resizable()
                .scaledToFit()
                .frame(width: 24, height: 24)

            Text("Penny")
                .font(.headline)

            Circle()
                .fill(viewModel.client.connectionColor)
                .frame(width: 9, height: 9)
                .accessibilityLabel(viewModel.client.statusText)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .accessibilityElement(children: .combine)
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

    private var bottomAnchorID: String { "message-list-bottom" }

    private var messageScrollCoordinateSpace: String { "message-scroll-coordinate-space" }

    private var messageRootCoordinateSpace: String { "message-root-coordinate-space" }

    private var bottomSpacer: some View {
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
    private func messageGrid(layout: MessageLayout) -> some View {
        Grid(horizontalSpacing: layout.itemSpacing, verticalSpacing: layout.itemSpacing) {
            ForEach(messageGridRows(for: layout)) { row in
                GridRow {
                    ForEach(row.messages) { message in
                        messageGridCell(message, layout: layout)
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

    private func messageGridCell(_ message: ChatMessage, layout: MessageLayout) -> some View {
        ChatMessageView(message: message, layout: layout)
            .id(message.id)
            .opacity(activeMessageContext?.message.id == message.id ? 0 : 1)
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .background {
                GeometryReader { geometry in
                    Color.clear.preference(
                        key: MessageFramePreferenceKey.self,
                        value: [message.id: geometry.frame(in: .named(messageRootCoordinateSpace))]
                    )
                }
            }
            .contentShape(Rectangle())
            .onTapGesture {
                if activeMessageContext?.message.id == message.id {
                    dismissMessageActions()
                    return
                }
                guard layout != .message else { return }
                presentedCardMessage = message
            }
            .onLongPressGesture(minimumDuration: 0.35) {
                presentMessageActions(for: message)
            }
            .animation(.spring(response: 0.24, dampingFraction: 0.78), value: activeMessageContext?.message.id)
    }

    private func messageGridRows(for layout: MessageLayout) -> [MessageGridRow] {
        let columnCount = layout.columnCount
        return stride(from: 0, to: viewModel.displayedMessages.count, by: columnCount).map { startIndex in
            let endIndex = min(startIndex + columnCount, viewModel.displayedMessages.count)
            return MessageGridRow(messages: Array(viewModel.displayedMessages[startIndex..<endIndex]))
        }
    }

    @ViewBuilder
    private func olderMessagesLoader(proxy: ScrollViewProxy) -> some View {
        if viewModel.isLoadingOlderMessages {
            ProgressView()
                .controlSize(.small)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
        }
    }

    private func topMessageLoader(proxy: ScrollViewProxy) -> some View {
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

}

private enum PennyNavigationDestination: String, CaseIterable, Identifiable {
    case schedules
    case insights
    case memories

    var id: Self { self }

    var title: String {
        switch self {
        case .schedules:
            return "Schedules"
        case .insights:
            return "Insights"
        case .memories:
            return "Memories"
        }
    }

    var systemImage: String {
        switch self {
        case .schedules:
            return "calendar"
        case .insights:
            return "chart.bar.doc.horizontal"
        case .memories:
            return "tray.full"
        }
    }
}

private extension MessageView {
    func refreshFeaturePreferences() {
        isMessageLayoutSwitcherEnabled = Prefs.shared.isMessageLayoutSwitcherEnabled
    }

    func handleComposerFocusChanged(_ isFocused: Bool) {
        guard isFocused else {
            shouldUseFastComposerScroll = false
            keyboardSettledScrollTask?.cancel()
            return
        }
        let didChangeMessageLayout = isMessageLayoutSwitcherEnabled && viewModel.prepareComposerFocus()
        scheduleScrollAfterKeyboardSettles(fast: shouldUseFastComposerScroll && !didChangeMessageLayout)
        shouldUseFastComposerScroll = false
    }

    func scheduleScrollAfterKeyboardSettles(fast: Bool = false) {
        keyboardSettledScrollTask?.cancel()
        keyboardSettledScrollTask = Task { @MainActor in
            try? await Task.sleep(for: fast ? .milliseconds(120) : .milliseconds(350))
            guard !Task.isCancelled && isComposerFocused else { return }
            viewModel.requestScrollToBottom(shouldSettleLayout: !fast)
        }
    }

    func changeMessageLayout(to layout: MessageLayout) {
        guard viewModel.selectedMessageLayout != layout else { return }
        if layout != .message {
            isComposerFocused = false
            keyboardSettledScrollTask?.cancel()
        }

        var transaction = Transaction(animation: .spring(response: 0.34, dampingFraction: 0.86))
        transaction.disablesAnimations = false
        withTransaction(transaction) {
            viewModel.selectedMessageLayout = layout
        }
        viewModel.requestScrollToBottom()
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
        if animated {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(bottomAnchorID, anchor: .bottom)
            }
        } else {
            proxy.scrollTo(bottomAnchorID, anchor: .bottom)
        }

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
        let fallbackProxyHeight = effectiveMessageLayout == .message ? context.frame.height : maximumProxyHeight
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
        guard effectiveMessageLayout != .message else {
            return min(max(1, context.frame.width), maximumWidth)
        }
        return maximumWidth
    }

    func actionProxyLeading(for context: MessageActionContext, width: CGFloat, in containerSize: CGSize) -> CGFloat {
        let maximumLeading = max(MessageLayout.message.horizontalPadding, containerSize.width - width - MessageLayout.message.horizontalPadding)
        if effectiveMessageLayout != .message {
            return context.message.isOutgoing ? maximumLeading : MessageLayout.message.horizontalPadding
        }
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
        return context.frame.height > maxHeight || (effectiveMessageLayout != .message && context.message.content.count > 280)
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
                shouldUseFastComposerScroll = true
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

private struct MessageFramePreferenceKey: PreferenceKey {
    static let defaultValue: [Int: CGRect] = [:]

    static func reduce(value: inout [Int: CGRect], nextValue: () -> [Int: CGRect]) {
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
        case .schedule:
            return "No scheduled messages"
        case .chat:
            return "No chat messages"
        case .notifier:
            return "No notifier messages"
        case .collector:
            return "No collector messages"
        }
    }
}

private struct TypingRow: View {
    var body: some View {
        HStack {
            HStack(spacing: 4) {
                ForEach(0..<3) { index in
                    Circle()
                        .fill(Color.secondary)
                        .frame(width: 6, height: 6)
                        .opacity(index == 1 ? 0.7 : 0.45)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))

            Spacer(minLength: 48)
        }
        .accessibilityLabel("Penny is typing")
    }
}

#Preview {
    MessageView()
}
