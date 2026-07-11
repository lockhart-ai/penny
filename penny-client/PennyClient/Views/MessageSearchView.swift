import SwiftUI

struct MessageSearchView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var selectedMessage: ChatMessage?
    @State private var selectedFilter: MessageView.MessageFilter = .all
    @State private var searchService: SearchService
    @State private var searchTask: Task<Void, Never>?

    init(client: PennyService) {
        _searchService = State(initialValue: SearchService(client: client))
    }

    private var searchResultColumns: [GridItem] {
        Array(
            repeating: GridItem(.flexible(), spacing: MessageView.MessageLayout.compact.itemSpacing, alignment: .top),
            count: MessageView.MessageLayout.compact.columnCount
        )
    }

    private var filterMenu: some View {
        Menu {
            Picker("Filter Messages", selection: $selectedFilter) {
                ForEach(MessageView.MessageFilter.allCases) { filter in
                    Label(filter.title, systemImage: filter.systemImage)
                        .tag(filter)
                }
            }
        } label: {
            Image(systemName: selectedFilter == .all ? "line.3.horizontal.decrease.circle" : "line.3.horizontal.decrease.circle.fill")
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.primary)
        .accessibilityLabel("Filter messages")
        .accessibilityValue(selectedFilter.title)
    }

    var body: some View {
        NavigationStack {
            Group {
                if searchService.isSearching {
                    ProgressView("Searching...")
                } else if let errorMessage = searchService.errorMessage {
                    ContentUnavailableView("Search Unavailable", systemImage: "exclamationmark.triangle", description: Text(errorMessage))
                } else if searchService.results.isEmpty {
                    ContentUnavailableView(
                        searchService.hasSearched ? "No Matches" : "Search Messages",
                        systemImage: "magnifyingglass",
                        description: Text(searchService.hasSearched ? "Try a different phrase." : "Search your conversation history by meaning.")
                    )
                } else {
                    ScrollView {
                        LazyVGrid(columns: searchResultColumns, spacing: MessageView.MessageLayout.compact.itemSpacing) {
                            ForEach(searchService.results) { result in
                                Button {
                                    selectedMessage = result.message
                                } label: {
                                    ChatMessageView(message: result.message, layout: .compact)
                                        .overlay(alignment: .bottomTrailing) {
                                            Text("\(Int(result.similarity * 100))%")
                                                .font(.caption2.weight(.semibold))
                                                .foregroundStyle(.secondary)
                                                .padding(.horizontal, 6)
                                                .padding(.vertical, 3)
                                                .background(.regularMaterial, in: Capsule())
                                                .padding(8)
                                        }
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(.horizontal, MessageView.MessageLayout.compact.horizontalPadding)
                        .padding(.vertical, 12)
                    }
                    .background(Color(.systemGroupedBackground))
                }
            }
            .navigationTitle("Search Messages")
            .navigationBarTitleDisplayMode(.inline)
            .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always), prompt: "Search by meaning")
            .onChange(of: query) { _, _ in
                cancelSearch()
                searchService.clear()
            }
            .onChange(of: selectedFilter) { _, _ in
                handleFilterChanged()
            }
            .onSubmit(of: .search) {
                startSearch()
            }
            .onDisappear {
                cancelSearch()
            }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    filterMenu
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Cancel") { dismiss() }
                }
            }
            .sheet(item: $selectedMessage) { message in
                MessageCardDetailSheet(message: message)
            }
        }
    }

    private func startSearch() {
        cancelSearch()
        searchTask = Task { await searchService.search(query, filter: selectedFilter.pageFilter) }
    }

    private func handleFilterChanged() {
        cancelSearch()
        guard searchService.hasSearched,
              !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            searchService.clear()
            return
        }
        startSearch()
    }

    private func cancelSearch() {
        searchTask?.cancel()
        searchTask = nil
    }
}
