import SwiftUI
import PennyDesign
import PennyServices

/// The Memory page's content under the app shell (#3111): today's memories
/// list; its pushed detail is `DataBrowserMemoryDetailView`, pushed by the
/// shell's own stack via `AppShellView.ShellRoute`. The shell owns the
/// model's lifecycle and connection handling; this view only renders the
/// list. Interim content — the redesign of this page has its own ticket.
struct DataBrowserMemoriesView: View {
    @Bindable var model: DataBrowserViewModel
    let reconnect: () -> Void

    var body: some View {
        memoriesList
            .overlay(alignment: .top) {
                DataBrowserOfflineBanner(model: model, reconnect: reconnect)
            }
            .onChange(of: model.context) { _, _ in model.contextChanged() }
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
                    .accessibilityIdentifier(AXID.memoryRow(memory.name))
            }
            if model.visibleMemories.isEmpty && !model.isDebouncing && !model.loading.contains(.memories) && model.errors[.memories] == nil && model.isOnline {
                ContentUnavailableView("No memories", systemImage: "tray", description: Text("No memories match this group and search."))
            }
        }
        .navigationTitle("Memories")
        .navigationBarTitleDisplayMode(.inline)
        .searchable(text: $model.memoryQuery, prompt: "Search memories and entries")
        .refreshable { model.refresh() }
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

/// A memory's pushed detail (#3111) — `AppShellView.ShellRoute.memoryDetail`'s
/// destination, with the system back button returning to
/// `DataBrowserMemoriesView`'s list.
struct DataBrowserMemoryDetailView: View {
    @Bindable var model: DataBrowserViewModel
    let name: String

    var body: some View {
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
        .refreshable { model.refresh() }
        .accessibilityIdentifier(AXID.memoryDetailRoot)
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

/// The Prompts page's content under the app shell (#3111): today's prompt
/// log list, including a message's linked run. The shell owns the model's
/// lifecycle and connection handling; this view only renders it. Interim
/// content — the redesign of this page has its own ticket.
struct DataBrowserPromptsView: View {
    @Bindable var model: DataBrowserViewModel
    let reconnect: () -> Void

    var body: some View {
        promptsList
            .overlay(alignment: .top) {
                DataBrowserOfflineBanner(model: model, reconnect: reconnect)
            }
            .onChange(of: model.context) { _, _ in model.contextChanged() }
    }

    @ViewBuilder private var promptsList: some View {
        if model.linkedRunID == nil {
            promptResults.searchable(text: $model.promptQuery, prompt: "Search prompts")
        } else {
            promptResults
        }
    }

    private var promptResults: some View {
        List {
            if model.linkedRunID != nil {
                Button("Browse all prompts") { model.linkedRunID = nil }
            } else {
                promptFilters
            }
            status(.prompts)
            ForEach(model.runs) { run in BrowserRunCard(run: run, model: model) }
            if model.runs.isEmpty && !model.isDebouncing && !model.loading.contains(.prompts) && model.errors[.prompts] == nil && model.isOnline {
                ContentUnavailableView("No prompts", systemImage: "text.bubble", description: Text(model.linkedRunID == nil ? "No runs match these filters." : "This message’s run is no longer available."))
            }
            if model.promptsHasMore {
                Button("Load more runs", action: model.loadMorePrompts)
                    .disabled(!model.isOnline || model.loading.contains(.prompts))
            }
        }
        .navigationTitle("Prompts")
        .navigationBarTitleDisplayMode(.inline)
        .refreshable { model.refresh() }
    }

    private var promptFilters: some View {
        Section("Filters") {
            TextField("Agent name (all when empty)", text: $model.agent)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Toggle("Flagged only", isOn: $model.flaggedOnly)
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

/// The disconnected banner both browser pages show at the top of their list.
private struct DataBrowserOfflineBanner: View {
    @Bindable var model: DataBrowserViewModel
    let reconnect: () -> Void

    var body: some View {
        if !model.isOnline {
            VStack(spacing: 8) {
                Label("Disconnected — reconnect to load data", systemImage: "wifi.slash")
                    .font(.callout)
                Button("Reconnect", action: reconnect)
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
