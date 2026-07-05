import SwiftUI
import UIKit

struct MessageView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var viewModel = ViewModel()
    @State private var isMessageLayoutSwitcherEnabled = Prefs.shared.isMessageLayoutSwitcherEnabled
    @State private var presentedCardMessage: ChatMessage?
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
            .overlay(alignment: .bottom) {
                composer
                    .offset(y: -viewModel.keyboardOffset)
                    .readHeight { height in
                        guard height > 0 else { return }
                        viewModel.composerHeight = height
                    }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 8) {
                        messageFilterMenu

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
            .ignoresSafeArea(.keyboard, edges: .bottom)
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
            .sheet(item: $presentedCardMessage) { message in
                MessageCardDetailSheet(message: message)
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
        .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillChangeFrameNotification)) { notification in
            viewModel.updateKeyboardHeight(from: notification)
        }
        .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillHideNotification)) { _ in
            viewModel.keyboardHeight = 0
        }
        .onDisappear {
            viewModel.disconnect()
        }
    }

    private var chatScrollView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 12) {
                    olderMessagesLoader(proxy: proxy)

                    if viewModel.displayedMessages.isEmpty {
                        EmptyMessageFilterView(filter: viewModel.selectedMessageFilter)
                    } else {
                        LazyVGrid(columns: MessageLayout.message.gridColumns, spacing: MessageLayout.message.itemSpacing) {
                            ForEach(viewModel.displayedMessages) { message in
                                ChatMessageView(message: message, layout: .message)
                                    .id(message.id)
                                    .onAppear {
                                        handleMessageAppeared(message, proxy: proxy)
                                    }
                            }
                        }
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
                scheduleScrollToBottom(with: proxy, animated: false)
            }
        }
    }

    private func cardScrollView(layout: MessageLayout) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 12) {
                    olderMessagesLoader(proxy: proxy)

                    if viewModel.displayedMessages.isEmpty {
                        EmptyMessageFilterView(filter: viewModel.selectedMessageFilter)
                    } else {
                        LazyVGrid(columns: layout.gridColumns, spacing: layout.itemSpacing) {
                            ForEach(viewModel.displayedMessages) { message in
                                ChatMessageView(message: message, layout: layout)
                                    .id(message.id)
                                    .contentShape(Rectangle())
                                    .onAppear {
                                        handleMessageAppeared(message, proxy: proxy)
                                    }
                                    .onTapGesture {
                                        presentedCardMessage = message
                                    }
                                }
                        }
                    }

                    bottomSpacer
                }
                .padding(.horizontal, layout.horizontalPadding)
                .padding(.top, isMessageLayoutSwitcherEnabled ? 58 : 12)
            }
            .background(Color(.systemGroupedBackground))
            .scrollDismissesKeyboard(.interactively)
            .onAppear {
                scheduleScrollToBottom(with: proxy, animated: false)
            }
            .onChange(of: viewModel.scrollToBottomRequest) { _, _ in
                scheduleScrollToBottom(with: proxy, animated: false)
            }
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
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.clear)
    }

    private var bottomAnchorID: String {
        "message-list-bottom"
    }

    private var bottomSpacer: some View {
        Color.clear
            .frame(height: viewModel.composerHeight + viewModel.keyboardOffset + 12)
            .id(bottomAnchorID)
            .onAppear {
                viewModel.updateBottomVisibility(true)
            }
            .onDisappear {
                viewModel.updateBottomVisibility(false)
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

    private func refreshFeaturePreferences() {
        isMessageLayoutSwitcherEnabled = Prefs.shared.isMessageLayoutSwitcherEnabled
    }

    private func changeMessageLayout(to layout: MessageLayout) {
        guard viewModel.selectedMessageLayout != layout else { return }

        var transaction = Transaction(animation: .spring(response: 0.34, dampingFraction: 0.86))
        transaction.disablesAnimations = false
        withTransaction(transaction) {
            viewModel.selectedMessageLayout = layout
        }
        viewModel.requestScrollToBottom()
    }

    private func scheduleScrollToBottom(
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

    private func scrollToBottom(
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

    private func loadOlderMessages(with proxy: ScrollViewProxy) {
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

    private func handleMessageAppeared(_ message: ChatMessage, proxy: ScrollViewProxy) {
        guard message.id == viewModel.displayedMessages.first?.id else { return }
        loadOlderMessages(with: proxy)
    }
}

private struct HeightPreferenceKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

private extension View {
    func readHeight(_ onChange: @escaping (CGFloat) -> Void) -> some View {
        overlay {
            GeometryReader { proxy in
                Color.clear
                    .preference(key: HeightPreferenceKey.self, value: proxy.size.height)
            }
        }
        .onPreferenceChange(HeightPreferenceKey.self, perform: onChange)
    }
}

private struct MessageCardDetailSheet: View {
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
