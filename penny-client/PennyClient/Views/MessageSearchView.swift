import Observation
import SwiftUI
import UIKit

struct MessageSearchView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var selectedMessage: ChatMessage?
    @State private var viewModel: MessageSearchViewModel

    private let onReply: (ChatMessage) -> Void

    init(client: PennyService, onReply: @escaping (ChatMessage) -> Void) {
        self.onReply = onReply
        _viewModel = State(initialValue: MessageSearchViewModel(client: client))
    }

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Search Messages")
                .navigationBarTitleDisplayMode(.inline)
                .searchable(
                    text: $query,
                    placement: .navigationBarDrawer(displayMode: .always),
                    prompt: "Search by meaning"
                )
                .onChange(of: query) { _, _ in
                    resetSearchAndLoadHistory()
                }
                .onChange(of: viewModel.selectedFilter) { _, _ in
                    filterChanged()
                }
                .onChange(of: viewModel.selectedLayout) { _, _ in
                    viewModel.resetHistoryPaging()
                }
                .onSubmit(of: .search) {
                    startSearch()
                }
                .onDisappear {
                    viewModel.cancelTasks()
                }
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        filterMenu
                    }
                    ToolbarItem(placement: .principal) {
                        layoutPicker
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Done") { dismiss() }
                    }
                }
                .sheet(item: $selectedMessage) { message in
                    MessageCardDetailSheet(message: message)
                }
        }
        .task {
            await viewModel.loadInitialHistory()
        }
    }

    @ViewBuilder
    private var content: some View {
        if viewModel.searchService.isSearching {
            ProgressView("Searching...")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if !trimmedQuery.isEmpty && viewModel.searchService.hasSearched {
            semanticResults
        } else {
            history
        }
    }

    private var history: some View {
        MessageCardGrid(
            messages: viewModel.displayedMessages,
            layout: viewModel.selectedLayout,
            onMessageTap: { selectedMessage = $0 },
            onReply: reply,
            onLastMessageAppear: loadMoreHistory
        )
        .overlay {
            if let errorMessage = viewModel.historyErrorMessage {
                ContentUnavailableView(
                    "History Unavailable",
                    systemImage: "exclamationmark.triangle",
                    description: Text(errorMessage)
                )
            } else if viewModel.displayedMessages.isEmpty && !viewModel.isLoadingHistory {
                ContentUnavailableView(
                    "No Messages",
                    systemImage: viewModel.selectedFilter.systemImage,
                    description: Text("No messages match the selected filter.")
                )
            } else if viewModel.isLoadingHistory {
                ProgressView()
            }
        }
    }

    @ViewBuilder
    private var semanticResults: some View {
        if let errorMessage = viewModel.searchService.errorMessage {
            ContentUnavailableView(
                "Search Unavailable",
                systemImage: "exclamationmark.triangle",
                description: Text(errorMessage)
            )
        } else if viewModel.searchService.results.isEmpty {
            ContentUnavailableView(
                "No Matches",
                systemImage: "magnifyingglass",
                description: Text("Try a different phrase.")
            )
        } else {
            MessageCardGrid(
                messages: viewModel.searchService.results.map(\.message),
                layout: viewModel.selectedLayout,
                onMessageTap: { selectedMessage = $0 },
                onReply: reply,
                badge: { message in
                    guard let result = viewModel.searchService.results.first(where: { $0.message.id == message.id }) else {
                        return nil
                    }
                    return "\(Int(result.similarity * 100))%"
                }
            )
        }
    }

    private var filterMenu: some View {
        Menu {
            Picker("Filter Messages", selection: $viewModel.selectedFilter) {
                ForEach(MessageView.MessageFilter.allCases) { filter in
                    Label(filter.title, systemImage: filter.systemImage)
                        .tag(filter)
                }
            }
        } label: {
            Image(systemName: viewModel.selectedFilter == .all
                ? "line.3.horizontal.decrease.circle"
                : "line.3.horizontal.decrease.circle.fill")
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.primary)
        .accessibilityLabel("Filter messages")
        .accessibilityValue(viewModel.selectedFilter.title)
    }

    private var layoutPicker: some View {
        Picker("Card layout", selection: $viewModel.selectedLayout) {
            Text("2").tag(MessageView.MessageLayout.compact)
            Text("3").tag(MessageView.MessageLayout.media)
        }
        .pickerStyle(.segmented)
        .frame(width: 96)
        .accessibilityLabel("Card layout")
    }

    private var trimmedQuery: String {
        query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func startSearch() {
        guard !trimmedQuery.isEmpty else {
            resetSearchAndLoadHistory()
            return
        }
        viewModel.cancelHistoryTask()
        Task {
            await viewModel.searchService.search(trimmedQuery, filter: viewModel.selectedFilter.pageFilter)
        }
    }

    private func resetSearchAndLoadHistory() {
        viewModel.searchService.clear()
        guard trimmedQuery.isEmpty else { return }
        Task {
            await viewModel.loadInitialHistory()
        }
    }

    private func filterChanged() {
        if trimmedQuery.isEmpty {
            Task { await viewModel.loadInitialHistory() }
        } else if viewModel.searchService.hasSearched {
            startSearch()
        }
    }

    private func loadMoreHistory() {
        guard trimmedQuery.isEmpty else { return }
        Task { await viewModel.loadMoreHistory() }
    }

    private func reply(to message: ChatMessage) {
        onReply(message)
        dismiss()
    }
}

@MainActor
@Observable
final class MessageSearchViewModel {
    let client: PennyService
    let searchService: SearchService
    var selectedFilter: MessageView.MessageFilter = .all
    var selectedLayout: MessageView.MessageLayout = .compact
    var displayedMessages: [ChatMessage] = []
    var isLoadingHistory = false
    var historyErrorMessage: String?

    @ObservationIgnored private let pageSize: Int
    @ObservationIgnored private var nextCursor: MessagePageCursor?
    @ObservationIgnored private var hasMoreHistory = false
    @ObservationIgnored private var historyTask: Task<Void, Never>?

    init(client: PennyService, pageSize: Int = 30, searchService: SearchService? = nil) {
        self.client = client
        self.pageSize = max(1, pageSize)
        self.searchService = searchService ?? SearchService(client: client)
    }

    var canLoadMoreHistory: Bool {
        hasMoreHistory && !isLoadingHistory
    }

    func loadInitialHistory() async {
        historyTask?.cancel()
        let task = Task { [weak self] in
            guard let self else { return }
            await self.loadInitialHistoryPage()
        }
        historyTask = task
        await task.value
    }

    func loadMoreHistory() async {
        guard canLoadMoreHistory, let nextCursor else { return }
        isLoadingHistory = true
        defer { isLoadingHistory = false }

        let page = await client.requestMessagePage(
            MessagePageRequest(limit: pageSize, before: nextCursor, filter: selectedFilter.pageFilter)
        )
        let existingIDs = Set(displayedMessages.map(\.id))
        displayedMessages.insert(contentsOf: page.messages.filter { !existingIDs.contains($0.id) }, at: 0)
        self.nextCursor = page.nextCursor
        hasMoreHistory = page.hasMore
    }

    func resetHistoryPaging() {
        nextCursor = nil
        hasMoreHistory = false
    }

    func cancelHistoryTask() {
        historyTask?.cancel()
        historyTask = nil
    }

    func cancelTasks() {
        cancelHistoryTask()
        searchService.clear()
    }

    private func loadInitialHistoryPage() async {
        isLoadingHistory = true
        historyErrorMessage = nil
        resetHistoryPaging()
        defer { isLoadingHistory = false }

        let page = await client.requestMessagePage(
            MessagePageRequest(limit: pageSize, filter: selectedFilter.pageFilter)
        )
        guard !Task.isCancelled else { return }
        displayedMessages = page.messages
        nextCursor = page.nextCursor
        hasMoreHistory = page.hasMore
    }
}

private struct MessageCardGrid: View {
    let messages: [ChatMessage]
    let layout: MessageView.MessageLayout
    let onMessageTap: (ChatMessage) -> Void
    let onReply: (ChatMessage) -> Void
    let badge: ((ChatMessage) -> String?)?
    let onLastMessageAppear: (() -> Void)?

    init(
        messages: [ChatMessage],
        layout: MessageView.MessageLayout,
        onMessageTap: @escaping (ChatMessage) -> Void,
        onReply: @escaping (ChatMessage) -> Void,
        badge: ((ChatMessage) -> String?)? = nil,
        onLastMessageAppear: (() -> Void)? = nil
    ) {
        self.messages = messages
        self.layout = layout
        self.onMessageTap = onMessageTap
        self.onReply = onReply
        self.badge = badge
        self.onLastMessageAppear = onLastMessageAppear
    }

    private var columns: [GridItem] {
        Array(repeating: GridItem(.flexible(), spacing: layout.itemSpacing, alignment: .top), count: layout.columnCount)
    }

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: layout.itemSpacing) {
                ForEach(messages) { message in
                    MessageCardCell(
                        message: message,
                        layout: layout,
                        badge: badge?(message),
                        onTap: { onMessageTap(message) },
                        onReply: { onReply(message) }
                    )
                    .onAppear {
                        guard message.id == messages.last?.id else { return }
                        onLastMessageAppear?()
                    }
                }
            }
            .padding(.horizontal, layout.horizontalPadding)
            .padding(.vertical, 12)
        }
        .background(Color(.systemGroupedBackground))
    }
}

private struct MessageCardCell: View {
    let message: ChatMessage
    let layout: MessageView.MessageLayout
    let badge: String?
    let onTap: () -> Void
    let onReply: () -> Void

    var body: some View {
        Button(action: onTap) {
            ChatMessageView(message: message, layout: layout)
                .overlay(alignment: .bottomTrailing) {
                    if let badge {
                        Text(badge)
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(.regularMaterial, in: Capsule())
                            .padding(8)
                    }
                }
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button(action: onReply) {
                Label("Reply", systemImage: "arrowshape.turn.up.left")
            }

            Button {
                UIPasteboard.general.string = message.content
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
        }
    }
}
