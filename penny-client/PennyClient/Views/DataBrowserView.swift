import SwiftUI
import PennyServices

struct DataBrowserView: View {
    let client: PennyService
    @State private var model: DataBrowserViewModel

    init(client: PennyService) {
        self.client = client
        _model = State(initialValue: DataBrowserViewModel { try await client.dataBrowser.read($0) })
    }

    var body: some View {
        DataBrowserContent(model: model, reconnect: client.reconnect)
            .onAppear { model.updateConnection(client.canSend) }
            .onChange(of: client.canSend) { _, connected in model.updateConnection(connected) }
            .onChange(of: client.browserMemoryRevision) { _, _ in
                if model.tab == .memories { model.scheduleRefresh() }
            }
            .onChange(of: client.browserRunRevision) { _, _ in
                if model.tab == .prompts || model.selection != nil { model.scheduleRefresh() }
            }
            .onDisappear { model.stop() }
    }
}

struct DataBrowserContent: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var model: DataBrowserViewModel
    let reconnect: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Picker("Browse", selection: $model.tab) {
                    ForEach(DataBrowserViewModel.Tab.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                Button("Close", systemImage: "xmark") { dismiss() }
                    .labelStyle(.iconOnly)
                    .frame(minWidth: 44, minHeight: 44)
                    .buttonStyle(.borderless)
            }
            .padding()
            if !model.isOnline {
                Label("Disconnected — reconnect to load data", systemImage: "wifi.slash")
                    .font(.callout)
                    .padding(.horizontal)
                Button("Reconnect", action: reconnect)
            }
            if model.tab == .memories {
                memoriesNavigation
            } else {
                NavigationStack { promptsList }
            }
        }
        .background(Color(.systemGroupedBackground))
        .frame(idealWidth: 600, idealHeight: 720)
        .onChange(of: model.context) { _, _ in model.contextChanged() }
    }

    private var memoriesNavigation: some View {
        NavigationStack(path: Binding(
            get: { model.selection.map { [$0] } ?? [] },
            set: { model.selection = $0.last }
        )) {
            memoriesList
                .navigationDestination(for: String.self) { name in memoryDetail(name) }
        }
    }

    private var memoriesList: some View {
        List {
            Picker("Memory group", selection: $model.group) {
                ForEach(DataBrowserViewModel.Group.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .onChange(of: model.group) { _, _ in model.changeGroup() }
            status(.memories)
            ForEach(model.visibleMemories) { memory in
                Button { model.select(memory) } label: { BrowserMemorySummary(memory: memory) }
                    .buttonStyle(.plain)
            }
            if model.visibleMemories.isEmpty && !model.isDebouncing && !model.loading.contains(.memories) && model.errors[.memories] == nil && model.isOnline {
                ContentUnavailableView("No memories", systemImage: "tray", description: Text("No memories match this group and search."))
            }
        }
        .navigationTitle("Memories")
        .navigationBarTitleDisplayMode(.inline)
        .searchable(text: $model.memoryQuery, prompt: "Search memories and entries")
        .toolbar { refreshButton }
    }

    private func memoryDetail(_ name: String) -> some View {
        List {
            Picker("Memory panel", selection: $model.panel) {
                ForEach(DataBrowserViewModel.Panel.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            status(.detail)
            if let detail = model.detail, detail.memory.name == name {
                switch model.panel {
                case .entries: entries(detail)
                case .activity: activity(detail)
                case .config: BrowserMemoryConfig(memory: detail.memory)
                }
            }
        }
        .navigationTitle(name)
        .navigationBarTitleDisplayMode(.inline)
        .searchable(text: $model.detailQuery, prompt: "Search entries")
        .toolbar { refreshButton }
    }

    @ViewBuilder
    private func entries(_ detail: MemoryDetail) -> some View {
        Section("Entries (\(detail.entries.count) shown of \(detail.memory.entryCount))") {
            if detail.entries.isEmpty {
                Text("No entries match.").foregroundStyle(.secondary)
            }
            ForEach(detail.entries) { entry in
                BrowserEntryCard(entry: entry, isLog: detail.memory.type == .log,
                                 expanded: model.expansion("entry:\(detail.memory.name):\(entry.id)"))
            }
            status(.entriesPage, retry: model.loadEntries)
            if detail.entriesHasMore {
                Button("Load more entries", action: model.loadEntries)
                    .disabled(!model.isOnline || model.loading.contains(.entriesPage) || model.loading.contains(.detail))
            }
        }
    }

    @ViewBuilder
    private func activity(_ detail: MemoryDetail) -> some View {
        if detail.memory.type == .log {
            Text("Logs are system-managed and have no collector activity.").foregroundStyle(.secondary)
        } else {
            if !detail.cursors.isEmpty {
                Section("Read cursors") {
                    Text("Where this collection has read up to in each input log.").font(.callout).foregroundStyle(.secondary)
                    ForEach(detail.cursors) { cursor in
                        VStack(alignment: .leading) {
                            Text(cursor.logName).font(.headline)
                            Text("Read up to \(BrowserDisplay.date(cursor.lastReadAt))").font(.caption)
                        }
                        .textSelection(.enabled)
                    }
                }
            }
            Section("Collector activity") {
                if detail.collectorRuns.isEmpty { Text("No collector activity yet.").foregroundStyle(.secondary) }
                ForEach(detail.collectorRuns) { run in BrowserRunCard(run: run, model: model) }
                status(.runsPage, retry: model.loadRuns)
                if detail.collectorRunsHasMore {
                    Button("Load more runs", action: model.loadRuns)
                        .disabled(!model.isOnline || model.loading.contains(.runsPage) || model.loading.contains(.detail))
                }
            }
        }
    }

    private var promptsList: some View {
        List {
            Section("Filters") {
                TextField("Agent name (all when empty)", text: $model.agent)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Toggle("Flagged only", isOn: $model.flaggedOnly)
            }
            status(.prompts)
            ForEach(model.runs) { run in BrowserRunCard(run: run, model: model) }
            if model.runs.isEmpty && !model.isDebouncing && !model.loading.contains(.prompts) && model.errors[.prompts] == nil && model.isOnline {
                ContentUnavailableView("No prompts", systemImage: "text.bubble", description: Text("No runs match these filters."))
            }
            if model.promptsHasMore {
                Button("Load more runs", action: model.loadMorePrompts)
                    .disabled(!model.isOnline || model.loading.contains(.prompts))
            }
        }
        .navigationTitle("Prompts")
        .navigationBarTitleDisplayMode(.inline)
        .searchable(text: $model.promptQuery, prompt: "Search prompts")
        .toolbar { refreshButton }
    }

    private var refreshButton: some ToolbarContent {
        ToolbarItem(placement: .topBarTrailing) {
            Button("Refresh", systemImage: "arrow.clockwise", action: model.refresh)
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
                .disabled(!model.isOnline || model.loading.contains(model.visibleSlot))
        }
    }

    @ViewBuilder
    private func status(_ slot: DataBrowserViewModel.Slot, retry: (() -> Void)? = nil) -> some View {
        if model.loading.contains(slot) || (slot == model.visibleSlot && model.isDebouncing) { ProgressView("Loading…") }
        if let error = model.errors[slot] {
            VStack(alignment: .leading, spacing: 8) {
                Text(error).foregroundStyle(.red)
                Button("Retry", action: retry ?? model.refresh).disabled(!model.isOnline)
            }
        }
    }
}

extension DataBrowserViewModel {
    func expansion(_ id: String) -> Binding<Bool> {
        Binding(get: { self.expanded.contains(id) }, set: { value in
            if value { self.expanded.insert(id) } else { self.expanded.remove(id) }
        })
    }
}
