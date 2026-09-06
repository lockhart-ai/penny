import Foundation
import Testing
@testable import PennyClient
@testable import PennyServices

@Suite(.serialized)
@MainActor
struct DataBrowserViewModelTests {
    @Test func groupsSelectionAndLiteralEntryPresentation() async throws {
        let harness = BrowserReadHarness()
        defer { harness.cancelAll() }
        let model = DataBrowserViewModel(read: harness.read)
        model.updateConnection(true)
        model.refresh()
        try await browserEventually { harness.requests.count == 1 }
        harness.resolve(0, .memories([memory("notes"), memory("events", type: .log), memory("old", archived: true)]))
        try await browserEventually { !model.loading.contains(.memories) }
        #expect(model.visibleMemories.map(\.name) == ["notes"])
        model.group = .logs
        #expect(model.visibleMemories.map(\.name) == ["events"])
        model.group = .archived
        #expect(model.visibleMemories.map(\.name) == ["old"])
        model.memoryQuery = "plain text"
        model.select(memory("notes"))
        #expect(model.selection == "notes")
        #expect(model.detailQuery == "plain text")
        #expect(model.panel == .entries)
        model.changeGroup()
        #expect(model.selection == nil)
        #expect(!BrowserDisplay.isLong(String(repeating: "a", count: 600)))
        #expect(BrowserDisplay.isLong(String(repeating: "a", count: 601)))
        #expect(BrowserDisplay.isLong(Array(repeating: "line", count: 21).joined(separator: "\n")))
        #expect(BrowserDisplay.text(.string("**literal**")) == "**literal**")
        #expect(BrowserDisplay.date("2026-09-01T12:00:00") == BrowserDisplay.date("2026-09-01T12:00:00Z"))
        model.stop()
    }

    @Test func pagesMergeIndependentlyAndAdvanceByReceivedRowCount() async throws {
        let harness = BrowserReadHarness()
        defer { harness.cancelAll() }
        let model = DataBrowserViewModel(read: harness.read)
        model.select(memory("notes"))
        model.updateConnection(true)
        model.refresh()
        try await browserEventually { harness.requests.count == 1 }
        harness.resolve(0, .detail(detail("notes", entries: [entry(1)], runs: [])))
        try await browserEventually { model.detail != nil }
        model.expanded = ["entry:notes:1"]
        model.panel = .activity
        model.loadEntries()
        model.loadEntries()
        model.loadRuns()
        try await browserEventually { harness.requests.count == 3 }
        harness.resolve(1, .page(page("notes", section: .entries, entries: [entry(1), entry(2)])))
        harness.resolve(2, .page(page("notes", section: .collectorRuns, entries: [])))
        try await browserEventually { model.loading.isEmpty }
        #expect(model.detail?.entries.map(\.id) == [1, 2])
        #expect(model.panel == .activity)
        #expect(model.expanded.contains("entry:notes:1"))
        model.loadEntries()
        try await browserEventually { harness.requests.count == 4 }
        guard case .page(_, .entries, let offset, _) = harness.requests[3] else { Issue.record("Expected entry page"); return }
        #expect(offset == 3)
        model.stop()
    }

    @Test func navigationAndSearchRejectStaleReplies() async throws {
        let harness = BrowserReadHarness()
        defer { harness.cancelAll() }
        let model = DataBrowserViewModel(read: harness.read)
        model.select(memory("first"))
        model.updateConnection(true)
        model.refresh()
        try await browserEventually { harness.requests.count == 1 }
        model.select(memory("second"))
        model.contextChanged()
        model.refresh()
        try await browserEventually { harness.requests.count == 2 }
        harness.resolve(1, .detail(detail("second", entries: [entry(2)], runs: [])))
        try await browserEventually { model.detail?.memory.name == "second" }
        harness.resolve(0, .detail(detail("first", entries: [entry(1)], runs: [])))
        await Task.yield()
        #expect(model.detail?.memory.name == "second")
        model.detailQuery = "new search"
        model.contextChanged()
        #expect(model.detail == nil)
        model.stop()
    }

    @Test func promptRefreshReplacesFilteredResultsAndFailurePreservesData() async throws {
        let harness = BrowserReadHarness()
        defer { harness.cancelAll() }
        let model = DataBrowserViewModel(read: harness.read)
        model.tab = .prompts
        model.updateConnection(true)
        model.refresh()
        try await browserEventually { harness.requests.count == 1 }
        harness.resolve(0, .prompts([try run("first")], hasMore: true))
        try await browserEventually { model.runs.count == 1 }
        model.loadMorePrompts()
        try await browserEventually { harness.requests.count == 2 }
        harness.resolve(1, .prompts([try run("first"), try run("second")], hasMore: true))
        try await browserEventually { model.runs.count == 2 }
        model.loadMorePrompts()
        try await browserEventually { harness.requests.count == 3 }
        guard case .prompts(_, _, _, let offset) = harness.requests[2] else { Issue.record("Expected prompts"); return }
        #expect(offset == 3)
        harness.fail(2)
        try await browserEventually { model.errors[.prompts] != nil }
        #expect(model.runs.count == 2)
        model.agent = "collector"
        model.flaggedOnly = true
        model.contextChanged()
        model.refresh()
        try await browserEventually { harness.requests.count == 4 }
        #expect(model.runs.isEmpty)
        harness.resolve(3, .prompts([try run("filtered")], hasMore: false))
        try await browserEventually { model.runs.first?.runID == "filtered" }
        model.stop()
    }

    @Test func eventsCoalesceAndDisconnectAndDismissCancelWork() async throws {
        let harness = BrowserReadHarness()
        defer { harness.cancelAll() }
        let model = DataBrowserViewModel(read: harness.read)
        model.updateConnection(true)
        for _ in 0..<10 { model.scheduleRefresh() }
        try await browserEventually { harness.requests.count == 1 }
        model.updateConnection(false)
        harness.resolve(0, .memories([memory("obsolete")]))
        await Task.yield()
        #expect(model.memories.isEmpty)
        #expect(model.loading.isEmpty)
        model.updateConnection(true)
        model.refresh()
        try await browserEventually { harness.requests.count == 2 }
        harness.resolve(1, .memories([memory("fresh")]))
        try await browserEventually { model.memories.count == 1 }
        model.refresh()
        try await browserEventually { harness.requests.count == 3 }
        model.stop()
        harness.resolve(2, .memories([]))
        await Task.yield()
        #expect(model.memories.first?.name == "fresh")
    }

    @Test func rapidSearchesDebounceAndEventsDuringReadRefreshTheFinalContext() async throws {
        let harness = BrowserReadHarness()
        defer { harness.cancelAll() }
        let model = DataBrowserViewModel(read: harness.read)
        model.tab = .prompts
        model.updateConnection(true)
        for query in ["p", "pr", "price"] {
            model.promptQuery = query
            model.contextChanged()
        }
        try await browserEventually { harness.requests.count == 1 }
        guard case .prompts(let agent, let query, let flagged, let offset) = harness.requests[0] else {
            Issue.record("Expected filtered prompt request"); return
        }
        #expect(agent == nil && query == "price" && !flagged && offset == 0)
        for _ in 0..<10 { model.scheduleRefresh() }
        // Let the event coalescer mark this in-flight read dirty, without cancelling it.
        try await Task.sleep(for: .milliseconds(350))
        #expect(harness.requests.count == 1)
        harness.resolve(0, .prompts([try run("first")], hasMore: false))
        try await browserEventually { harness.requests.count == 2 }
        #expect(model.runs.first?.id == "first")
        harness.resolve(1, .prompts([try run("updated")], hasMore: false))
        try await browserEventually { model.loading.isEmpty }
        #expect(model.runs.map(\.id) == ["updated"])
        model.stop()
    }

    @Test func detailRefreshReplacesPagesAndRetryKeepsPanelsAndExpansion() async throws {
        let harness = BrowserReadHarness()
        defer { harness.cancelAll() }
        let model = DataBrowserViewModel(read: harness.read)
        model.select(memory("notes"))
        model.updateConnection(true)
        model.refresh()
        try await browserEventually { harness.requests.count == 1 }
        harness.resolve(0, .detail(detail("notes", entries: [entry(1)], runs: [try run("old")])))
        try await browserEventually { model.detail != nil }
        model.panel = .activity
        model.expanded = ["run:old"]
        model.loadRuns()
        try await browserEventually { harness.requests.count == 2 }
        guard case .page(_, .collectorRuns, let offset, _) = harness.requests[1] else {
            Issue.record("Expected run page"); return
        }
        #expect(offset == 1)
        harness.resolve(1, .page(page("notes", section: .collectorRuns, entries: [], runs: [try run("old"), try run("next")])))
        try await browserEventually { model.loading.isEmpty }
        #expect(model.detail?.collectorRuns.map(\.id) == ["old", "next"])
        model.refresh()
        try await browserEventually { harness.requests.count == 3 }
        harness.fail(2)
        try await browserEventually { model.errors[.detail] != nil }
        #expect(model.detail?.collectorRuns.count == 2)
        model.refresh()
        try await browserEventually { harness.requests.count == 4 }
        harness.resolve(3, .detail(detail("notes", entries: [entry(2)], runs: [try run("old")])))
        try await browserEventually { model.loading.isEmpty }
        #expect(model.detail?.entries.map(\.id) == [2])
        #expect(model.detail?.collectorRuns.map(\.id) == ["old"])
        #expect(model.panel == .activity && model.expanded.contains("run:old"))
        model.stop()
    }
}

@MainActor
private final class BrowserReadHarness {
    var requests: [DataBrowserRequest] = []
    private var continuations: [Int: CheckedContinuation<DataBrowserResult, Error>] = [:]

    func read(_ request: DataBrowserRequest) async throws -> DataBrowserResult {
        let index = requests.count
        requests.append(request)
        return try await withCheckedThrowingContinuation { continuations[index] = $0 }
    }

    func resolve(_ index: Int, _ result: DataBrowserResult) { continuations.removeValue(forKey: index)?.resume(returning: result) }
    func fail(_ index: Int) { continuations.removeValue(forKey: index)?.resume(throwing: DataBrowserError.timedOut) }
    func cancelAll() {
        for continuation in continuations.values { continuation.resume(throwing: CancellationError()) }
        continuations.removeAll()
    }
}

@MainActor
private func browserEventually(_ condition: () -> Bool) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(3))
    while !condition() {
        guard ContinuousClock.now < deadline else { throw DataBrowserError.timedOut }
        await Task.yield()
    }
}

private func memory(_ name: String, type: MemoryType = .collection, archived: Bool = false) -> MemoryRecord {
    MemoryRecord(name: name, type: type, description: "Synthetic notes", notificationsEnabled: true, archived: archived,
                 extractionPrompt: nil, schedule: nil, lastCollectedAt: nil, entryCount: 1)
}

private func entry(_ identifier: Int) -> MemoryEntryRecord {
    MemoryEntryRecord(id: identifier, key: "key-\(identifier)", content: "**literal**", author: "fixture", createdAt: "2026-09-01T12:00:00Z")
}

private func detail(_ name: String, entries: [MemoryEntryRecord], runs: [PromptLogRun]) -> MemoryDetail {
    MemoryDetail(payload: MemoryDetailResponsePayload(memory: memory(name), entries: entries, entriesHasMore: true,
                                                     collectorRuns: runs, collectorRunsHasMore: true, cursors: []))
}

private func page(_ name: String, section: MemorySection, entries: [MemoryEntryRecord], runs: [PromptLogRun] = []) -> MemoryPage {
    MemoryPage(payload: MemoryPageResponsePayload(name: name, section: section, entries: entries, runs: runs, hasMore: true))
}

private func run(_ identifier: String) throws -> PromptLogRun {
    let json = """
    {"run_id":"\(identifier)","agent_name":"collector","prompt_count":0,"started_at":"2026-09-01T12:00:00Z",
    "ended_at":"2026-09-01T12:00:01Z","total_duration_ms":1000,"total_input_tokens":0,"total_output_tokens":0,
    "run_outcome":"worked","run_reason":"Saved an entry","run_target":"notes","record":"Saved an entry.","prompts":[],
    "health":{"bailed":false,"no_writes":false,"incomplete":false,"tool_failures":0,"degenerate_send":false,"flags":[],"regressive":false}}
    """
    return try JSONDecoder().decode(PromptLogRun.self, from: Data(json.utf8))
}
