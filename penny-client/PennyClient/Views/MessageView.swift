import SwiftUI
import UIKit

struct MessageView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var viewModel = ViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(viewModel.client.messages) { message in
                                ChatMessageRow(message: message)
                                    .id(message.id)
                            }

                            if viewModel.client.isTyping {
                                TypingRow()
                            }

                            Color.clear
                                .frame(height: viewModel.composerHeight + viewModel.keyboardOffset + 12)
                                .id(bottomAnchorID)
                        }
                        .padding(.horizontal, 16)
                        .padding(.top, 12)
                    }
                    .background(Color(.systemGroupedBackground))
                    .scrollDismissesKeyboard(.interactively)
                    .onAppear {
                        DispatchQueue.main.async {
                            scrollToBottom(with: proxy, animated: false)
                        }
                    }
                    .onChange(of: viewModel.client.messages.count) { _, _ in
                        scrollToBottom(with: proxy)
                    }
                    .onChange(of: viewModel.client.isTyping) { _, _ in
                        scrollToBottom(with: proxy)
                    }
                    .onChange(of: viewModel.keyboardHeight) { _, _ in
                        scrollToBottom(with: proxy)
                    }
                }
            }
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
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
        }
        .task {
            await viewModel.connect()
        }
        .onChange(of: scenePhase) { _, newPhase in
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

    private func scrollToBottom(with proxy: ScrollViewProxy, animated: Bool = true) {
        if animated {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(bottomAnchorID, anchor: .bottom)
            }
        } else {
            proxy.scrollTo(bottomAnchorID, anchor: .bottom)
        }
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

private struct ChatMessageRow: View {
    let message: ChatMessage

    private var markdownTextBlocks: [AttributedString] {
        message.content
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { line in
                let text = line.isEmpty ? " " : String(line)
                return (try? AttributedString(markdown: text)) ?? AttributedString(text)
            }
    }

    @ViewBuilder
    private var messageBubble: some View {
        if message.isOutgoing {
            Text(message.content)
                .font(.body)
                .foregroundStyle(.white)
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
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
    }

    var body: some View {
        HStack {
            if message.isOutgoing {
                Spacer(minLength: 48)
            }

            VStack(alignment: message.isOutgoing ? .trailing : .leading, spacing: 5) {
                if let sourceHint = message.sourceHint, !sourceHint.isEmpty {
                    Text(sourceHint)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }

                messageBubble

                Text(message.displayTime)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if !message.isOutgoing {
                Spacer(minLength: 48)
            }
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
