import Foundation
import Observation
import PennyServices

@MainActor
@Observable
final class DataBrowserViewModel {
    enum Tab: String, CaseIterable { case memories = "Memories", prompts = "Prompts" }
    enum Group: String, CaseIterable { case collections = "Collections", logs = "Logs", archived = "Archived" }
    enum Panel: String, CaseIterable { case entries = "Entries", activity = "Activity", config = "Config" }
    enum Slot: Hashable { case memories, detail, prompts, entriesPage, runsPage }

    var tab: Tab = .memories
    var group: Group = .collections
    var panel: Panel = .entries
    var selection: String?
    var memoryQuery = ""
    var detailQuery = ""
    var promptQuery = ""
    var agent = ""
    var flaggedOnly = false
    var expanded: Set<String> = []
    var recordRuns: Set<String> = []
    private(set) var memories: [MemoryRecord] = []
    private(set) var detail: MemoryDetail?
    private(set) var runs: [PromptLogRun] = []
    private(set) var promptsHasMore = false
    private(set) var loading: Set<Slot> = []
    private(set) var errors: [Slot: String] = [:]
    private(set) var isOnline = false
    private(set) var isDebouncing = false
    private var entryOffset = 0
    private var runOffset = 0
    private var promptOffset = 0
    @ObservationIgnored private let read: @MainActor (DataBrowserRequest) async throws -> DataBrowserResult
    @ObservationIgnored private var tasks: [Slot: Task<Void, Never>] = [:]
    @ObservationIgnored private var generations: [Slot: UUID] = [:]
    @ObservationIgnored private var refreshTask: Task<Void, Never>?
    @ObservationIgnored private var lastContext: Context?
    @ObservationIgnored private var refreshPending = false

    struct Context: Equatable {
        let tab: Tab
        let group: Group
        let selection: String?
        let memoryQuery: String
        let detailQuery: String
        let promptQuery: String
        let agent: String
        let flagged: Bool
    }

    var context: Context {
        Context(tab: tab, group: group, selection: selection, memoryQuery: memoryQuery,
                detailQuery: detailQuery, promptQuery: promptQuery, agent: agent, flagged: flaggedOnly)
    }

    var visibleMemories: [MemoryRecord] {
        memories.filter { memory in
            switch group {
            case .archived: return memory.archived
            case .collections: return !memory.archived && memory.type == .collection
            case .logs: return !memory.archived && memory.type == .log
            }
        }
    }

    var visibleSlot: Slot { tab == .prompts ? .prompts : selection == nil ? .memories : .detail }

    init(read: @escaping @MainActor (DataBrowserRequest) async throws -> DataBrowserResult) {
        self.read = read
    }

    func select(_ memory: MemoryRecord) {
        selection = memory.name
        detailQuery = memoryQuery
        detail = nil
        panel = .entries
    }

    func changeGroup() {
        selection = nil
        detail = nil
    }

    func updateConnection(_ online: Bool) {
        isOnline = online
        cancelRequests()
        if online { contextChanged() }
    }

    func contextChanged() {
        let previous = lastContext
        lastContext = context
        cancelRequests()
        if previous?.memoryQuery != memoryQuery { memories = [] }
        if previous?.selection != selection || previous?.detailQuery != detailQuery { detail = nil }
        if previous?.promptQuery != promptQuery || previous?.agent != agent || previous?.flagged != flaggedOnly {
            runs = []
            promptsHasMore = false
        }
        scheduleRefresh()
    }

    func scheduleRefresh() {
        refreshTask?.cancel()
        guard isOnline else { return }
        isDebouncing = true
        refreshTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .milliseconds(250))
                try Task.checkCancellation()
                guard let self else { return }
                isDebouncing = false
                if loading.contains(visibleSlot) { refreshPending = true } else { refresh() }
            } catch { /* A newer context or event superseded this refresh. */ }
        }
    }

    func refresh() {
        guard isOnline else { return }
        isDebouncing = false
        refreshTask?.cancel()
        cancel(.entriesPage)
        cancel(.runsPage)
        switch tab {
        case .memories:
            if let selection {
                start(.detail, request: .detail(name: selection, query: trimmed(detailQuery))) { model, result in
                    guard case .detail(let detail) = result, detail.memory.name == model.selection else { throw DataBrowserError.invalidResponse }
                    var normalized = detail
                    normalized.entries = Self.merge([], detail.entries)
                    normalized.collectorRuns = Self.merge([], detail.collectorRuns)
                    model.detail = normalized
                    model.entryOffset = detail.entries.count
                    model.runOffset = detail.collectorRuns.count
                }
            } else {
                start(.memories, request: .memories(query: trimmed(memoryQuery))) { model, result in
                    guard case .memories(let memories) = result else { throw DataBrowserError.invalidResponse }
                    model.memories = memories
                }
            }
        case .prompts:
            loadPrompts(append: false)
        }
    }

    func loadEntries() {
        guard let selection, let detail, detail.entriesHasMore,
              !loading.contains(.detail), !loading.contains(.entriesPage) else { return }
        start(.entriesPage, request: .page(name: selection, section: .entries, offset: entryOffset, query: trimmed(detailQuery))) { model, result in
            guard case .page(let page) = result, page.name == model.selection, page.section == .entries else { throw DataBrowserError.invalidResponse }
            guard var detail = model.detail else { throw DataBrowserError.invalidResponse }
            model.entryOffset += page.entries.count
            detail.entries = Self.merge(detail.entries, page.entries)
            detail.entriesHasMore = page.hasMore
            model.detail = detail
        }
    }

    func loadRuns() {
        guard let selection, let detail, detail.collectorRunsHasMore,
              !loading.contains(.detail), !loading.contains(.runsPage) else { return }
        start(.runsPage, request: .page(name: selection, section: .collectorRuns, offset: runOffset, query: nil)) { model, result in
            guard case .page(let page) = result, page.name == model.selection, page.section == .collectorRuns else { throw DataBrowserError.invalidResponse }
            guard var detail = model.detail else { throw DataBrowserError.invalidResponse }
            model.runOffset += page.runs.count
            detail.collectorRuns = Self.merge(detail.collectorRuns, page.runs)
            detail.collectorRunsHasMore = page.hasMore
            model.detail = detail
        }
    }

    func loadMorePrompts() {
        guard promptsHasMore, !loading.contains(.prompts) else { return }
        loadPrompts(append: true)
    }

    private func loadPrompts(append: Bool) {
        start(.prompts, request: .prompts(agent: trimmed(agent), query: trimmed(promptQuery), flagged: flaggedOnly, offset: append ? promptOffset : 0)) { model, result in
            guard case .prompts(let runs, let hasMore) = result else { throw DataBrowserError.invalidResponse }
            model.promptOffset = (append ? model.promptOffset : 0) + runs.count
            model.runs = Self.merge(append ? model.runs : [], runs)
            model.promptsHasMore = hasMore
        }
    }

    func stop() {
        cancelRequests()
        isOnline = false
    }

    private func start(_ slot: Slot, request: DataBrowserRequest,
                       apply: @escaping @MainActor (DataBrowserViewModel, DataBrowserResult) throws -> Void) {
        guard isOnline else { return }
        cancel(slot)
        let generation = UUID()
        generations[slot] = generation
        loading.insert(slot)
        errors[slot] = nil
        tasks[slot] = Task { [weak self, read] in
            do {
                let result = try await read(request)
                try Task.checkCancellation()
                guard let self, generations[slot] == generation else { return }
                try apply(self, result)
                complete(slot)
            } catch {
                guard let self, generations[slot] == generation else { return }
                if !(error is CancellationError) { errors[slot] = error.localizedDescription }
                complete(slot)
            }
        }
    }

    private func complete(_ slot: Slot) {
        loading.remove(slot)
        tasks[slot] = nil
        generations[slot] = nil
        if refreshPending && slot == visibleSlot {
            refreshPending = false
            scheduleRefresh()
        }
    }

    private func cancel(_ slot: Slot) {
        generations[slot] = nil
        tasks.removeValue(forKey: slot)?.cancel()
        loading.remove(slot)
        errors[slot] = nil
    }

    private func cancelRequests() {
        isDebouncing = false
        refreshPending = false
        refreshTask?.cancel()
        refreshTask = nil
        for slot in Array(tasks.keys) { cancel(slot) }
    }

    private func trimmed(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func merge<Item: Identifiable>(_ existing: [Item], _ incoming: [Item]) -> [Item] where Item.ID: Hashable {
        var merged = existing
        var positions = Dictionary(uniqueKeysWithValues: existing.enumerated().map { ($0.element.id, $0.offset) })
        for item in incoming {
            if let position = positions[item.id] { merged[position] = item } else {
                positions[item.id] = merged.count
                merged.append(item)
            }
        }
        return merged
    }
}
