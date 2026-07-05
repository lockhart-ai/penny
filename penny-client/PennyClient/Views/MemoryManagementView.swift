import SwiftUI

struct MemoryManagementView: View {
    @State private var viewModel: MemoryManagementViewModel
    @State private var editedDescription = ""
    @State private var editedIntent = ""
    @State private var editedInclusion: MemoryInclusion = .relevant
    @State private var editedRecall: MemoryRecall = .recent
    @State private var editedPublished = false

    init(client: PennyWebSocketClient) {
        _viewModel = State(initialValue: MemoryManagementViewModel(client: client))
    }

    var body: some View {
        Form {
            Section("Find") {
                TextField("Search memories", text: $viewModel.query)
                Button {
                    viewModel.refresh()
                } label: {
                    Label("Search", systemImage: "magnifyingglass")
                }
            }

            Section("Create") {
                TextField("Name", text: $viewModel.newName)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                TextField("Description", text: $viewModel.newDescription, axis: .vertical)
                    .lineLimit(1...3)
                TextField("Intent", text: $viewModel.newIntent, axis: .vertical)
                    .lineLimit(1...3)
                Picker("Inclusion", selection: $viewModel.newInclusion) {
                    ForEach(MemoryInclusion.allCasesForUI, id: \.self) { inclusion in
                        Text(inclusion.title).tag(inclusion)
                    }
                }
                Picker("Recall", selection: $viewModel.newRecall) {
                    ForEach(MemoryRecall.allCasesForUI, id: \.self) { recall in
                        Text(recall.title).tag(recall)
                    }
                }
                Toggle("Published", isOn: $viewModel.newPublished)
                TextField("Extraction prompt", text: $viewModel.newExtractionPrompt, axis: .vertical)
                    .lineLimit(1...4)
                TextField("Collector interval seconds", text: $viewModel.newCollectorIntervalText)
                    .keyboardType(.numberPad)
                Button {
                    viewModel.createMemory()
                } label: {
                    Label("Create", systemImage: "plus")
                }
                .disabled(!viewModel.canCreateMemory)
            }

            Section("Memories") {
                if viewModel.memories.isEmpty {
                    ContentUnavailableView("No Memories", systemImage: "tray")
                } else {
                    ForEach(viewModel.memories) { memory in
                        Button {
                            select(memory)
                        } label: {
                            MemorySummaryRow(memory: memory, isSelected: memory.name == viewModel.selectedMemoryName)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            if let memory = viewModel.selectedMemory {
                Section("Selected") {
                    MemorySummaryRow(memory: memory, isSelected: true)
                    TextField("Description", text: $editedDescription, axis: .vertical)
                        .lineLimit(1...4)
                    TextField("Intent", text: $editedIntent, axis: .vertical)
                        .lineLimit(1...4)
                    Picker("Inclusion", selection: $editedInclusion) {
                        ForEach(MemoryInclusion.allCasesForUI, id: \.self) { inclusion in
                            Text(inclusion.title).tag(inclusion)
                        }
                    }
                    Picker("Recall", selection: $editedRecall) {
                        ForEach(MemoryRecall.allCasesForUI, id: \.self) { recall in
                            Text(recall.title).tag(recall)
                        }
                    }
                    Toggle("Published", isOn: $editedPublished)
                    Button {
                        viewModel.updateSelectedMemory(
                            description: editedDescription,
                            intent: editedIntent,
                            inclusion: editedInclusion,
                            recall: editedRecall,
                            published: editedPublished
                        )
                    } label: {
                        Label("Save Memory", systemImage: "checkmark")
                    }
                    Button {
                        viewModel.triggerCollection()
                    } label: {
                        Label("Trigger Collection", systemImage: "arrow.triangle.2.circlepath")
                    }
                    Button(role: .destructive) {
                        viewModel.archiveSelectedMemory()
                    } label: {
                        Label("Archive", systemImage: "archivebox")
                    }
                }
            }

            if let detail = viewModel.detail {
                Section("Entries") {
                    TextField("Key", text: $viewModel.entryKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    TextField("Content", text: $viewModel.entryContent, axis: .vertical)
                        .lineLimit(1...5)
                    Button {
                        viewModel.submitEntry()
                    } label: {
                        Label("Add Entry", systemImage: "plus")
                    }
                    .disabled(!viewModel.canSubmitEntry)

                    ForEach(detail.entries) { entry in
                        MemoryEntryRow(entry: entry, viewModel: viewModel)
                    }

                    if detail.entriesHasMore {
                        Button {
                            viewModel.loadMoreEntries()
                        } label: {
                            Label("Load More Entries", systemImage: "ellipsis.circle")
                        }
                    }
                }

                Section("Collector Runs") {
                    ForEach(detail.collectorRuns) { run in
                        LabeledContent(run.agentName, value: run.runTarget ?? run.startedAt)
                    }

                    if detail.collectorRunsHasMore {
                        Button {
                            viewModel.loadMoreCollectorRuns()
                        } label: {
                            Label("Load More Runs", systemImage: "ellipsis.circle")
                        }
                    }
                }

                Section("Cursors") {
                    ForEach(detail.cursors) { cursor in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(cursor.logName)
                                Text(cursor.lastReadAt)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button(role: .destructive) {
                                viewModel.clearCursor(logName: cursor.logName)
                            } label: {
                                Image(systemName: "xmark.circle")
                            }
                            .buttonStyle(.borderless)
                            .accessibilityLabel("Clear cursor")
                        }
                    }
                }
            }
        }
        .navigationTitle("Memory")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    viewModel.refresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .accessibilityLabel("Refresh memories")
            }
        }
        .task {
            viewModel.refresh()
        }
    }

    private func select(_ memory: MemoryRecord) {
        viewModel.select(memory: memory)
        editedDescription = memory.description
        editedIntent = memory.intent ?? ""
        editedInclusion = memory.inclusion
        editedRecall = memory.recall
        editedPublished = memory.published
    }
}

private struct MemorySummaryRow: View {
    let memory: MemoryRecord
    let isSelected: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: memory.type == .collection ? "tray.full" : "doc.text")
                .foregroundStyle(isSelected ? Color.accentColor : .secondary)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 4) {
                Text(memory.name)
                    .font(.headline)
                Text(memory.description)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                HStack {
                    Text(memory.inclusion.title)
                    Text(memory.recall.title)
                    Text("\(memory.entryCount) entries")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 3)
    }
}

private struct MemoryEntryRow: View {
    let entry: MemoryEntryRecord
    let viewModel: MemoryManagementViewModel
    @State private var content: String

    init(entry: MemoryEntryRecord, viewModel: MemoryManagementViewModel) {
        self.entry = entry
        self.viewModel = viewModel
        _content = State(initialValue: entry.content)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(entry.key ?? "Entry \(entry.id)")
                .font(.headline)
            TextField("Content", text: $content, axis: .vertical)
                .lineLimit(1...5)
            HStack {
                Button {
                    viewModel.updateEntry(key: entry.key, content: content)
                } label: {
                    Label("Save", systemImage: "checkmark")
                }

                Spacer()

                Button(role: .destructive) {
                    viewModel.deleteEntry(key: entry.key)
                } label: {
                    Label("Delete", systemImage: "trash")
                }
            }
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 4)
    }
}

private extension MemoryInclusion {
    static let allCasesForUI: [MemoryInclusion] = [.always, .relevant, .never]

    var title: String {
        switch self {
        case .always:
            return "Always"
        case .relevant:
            return "Relevant"
        case .never:
            return "Never"
        }
    }
}

private extension MemoryRecall {
    static let allCasesForUI: [MemoryRecall] = [.all, .relevant, .recent]

    var title: String {
        switch self {
        case .all:
            return "All"
        case .relevant:
            return "Relevant"
        case .recent:
            return "Recent"
        }
    }
}

#Preview {
    NavigationStack {
        MemoryManagementView(client: PennyWebSocketClient())
    }
}
