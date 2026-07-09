import Foundation
import Testing
@testable import PennyClient

@Suite(.serialized)
@MainActor
struct PennyAdminViewModelTests {
    @Test func appBuildInfoReadsCommitHashOnlyWhenPresent() {
        #expect(AppBuildInfo(infoDictionary: nil).commitHash == nil)
        #expect(AppBuildInfo(infoDictionary: ["PennyBuildCommitHash": "   "]).commitHash == nil)
        #expect(AppBuildInfo(infoDictionary: ["PennyBuildCommitHash": " abc123def456 \n"]).commitHash == "abc123def456")
    }

    @Test func schedulesViewModelRefreshesAddsUpdatesAndDeletes() async throws {
        let (client, transport) = makeAdminClient()
        await connectAndClearStartupFrames(client, transport)
        let viewModel = SchedulesViewModel(client: client)

        viewModel.refresh()
        viewModel.draftCommand = "  summarize unread newsletters  "
        viewModel.addSchedule()

        transport.emit("""
        {
          "type": "schedules_response",
          "error": null,
          "schedules": [{
            "id": 42,
            "timing_description": "every morning",
            "prompt_text": "old prompt",
            "cron_expression": "0 8 * * *"
          }]
        }
        """)
        let schedule = try #require(viewModel.schedules.first)
        viewModel.setPromptText("  new prompt  ", for: schedule)
        viewModel.save(schedule: schedule)
        viewModel.delete(schedule: schedule)

        let payloads = await sentPayloads(transport, count: 4)
        #expect(payloads.map(typeName) == [
            "schedules_request",
            "schedule_add",
            "schedule_update",
            "schedule_delete"
        ])
        #expect(payloads[1]["command"] == .string("summarize unread newsletters"))
        #expect(payloads[2]["schedule_id"] == .number(42))
        #expect(payloads[2]["prompt_text"] == .string("new prompt"))
        #expect(payloads[3]["schedule_id"] == .number(42))
        #expect(viewModel.draftCommand.isEmpty)
        client.disconnect()
    }

    @Test func insightsViewModelRequestsFiltersAndLoadMore() async throws {
        let (client, transport) = makeAdminClient()
        await connectAndClearStartupFrames(client, transport)
        let viewModel = InsightsViewModel(client: client)
        viewModel.selectedAgentName = " collector "
        viewModel.query = " failed "
        viewModel.flaggedOnly = true

        viewModel.refresh()
        transport.emit(promptLogsResponseJSON(hasMore: true))
        viewModel.loadMore()

        let payloads = await sentPayloads(transport, count: 2)
        #expect(payloads[0]["type"] == .string("prompt_logs_request"))
        #expect(payloads[0]["agent_name"] == .string("collector"))
        #expect(payloads[0]["query"] == .string("failed"))
        #expect(payloads[0]["flagged_only"] == .bool(true))
        #expect(payloads[0]["offset"] == nil)
        #expect(payloads[1]["offset"] == .number(2))
        #expect(viewModel.runs.count == 2)
        #expect(viewModel.totalPromptCount == 3)
        #expect(viewModel.failedRunCount == 1)
        #expect(viewModel.requestedOffset == 2)
        client.disconnect()
    }

    @Test func memoryManagementViewModelCoversListDetailMutationAndPagination() async throws {
        let (client, transport) = makeAdminClient()
        await connectAndClearStartupFrames(client, transport)
        let viewModel = MemoryManagementViewModel(client: client)
        viewModel.query = " recipes "

        viewModel.refresh()
        transport.emit("""
        {
          "type": "memories_response",
          "memories": [\(memoryRecordJSON(name: "recipes"))]
        }
        """)
        let memory = try #require(viewModel.memories.first)
        viewModel.select(memory: memory)
        transport.emit("""
        {
          "type": "memory_detail_response",
          "memory": \(memoryRecordJSON(name: "recipes")),
          "entries": [{
            "id": 7,
            "key": "pasta",
            "content": "likes carbonara",
            "author": "user",
            "created_at": "2026-07-05T12:00:00Z"
          }],
          "entries_has_more": true,
          "collector_runs": [],
          "collector_runs_has_more": true,
          "cursors": [{
            "log_name": "penny.log",
            "last_read_at": "2026-07-05T12:00:00Z"
          }]
        }
        """)

        viewModel.newName = " meals "
        viewModel.newDescription = " meal notes "
        viewModel.newIntent = " remember food "
        viewModel.newInclusion = .always
        viewModel.newRecall = .all
        viewModel.newPublished = true
        viewModel.newExtractionPrompt = " extract meals "
        viewModel.newCollectorIntervalText = "3600"
        viewModel.createMemory()
        viewModel.loadMoreEntries()
        viewModel.loadMoreCollectorRuns()
        viewModel.entryKey = " dinner "
        viewModel.entryContent = "  likes soup  "
        viewModel.submitEntry()
        viewModel.updateEntry(key: "pasta", content: "likes cacio e pepe")
        viewModel.deleteEntry(key: "pasta")
        viewModel.triggerCollection()
        viewModel.setCursor(logName: "penny.log", lastReadAt: "2026-07-05T13:00:00Z")
        viewModel.clearCursor(logName: "penny.log")
        viewModel.archiveSelectedMemory()

        let payloads = await sentPayloads(transport, count: 12)
        #expect(payloads.map(typeName) == [
            "memories_request",
            "memory_detail_request",
            "memory_create",
            "memory_page_request",
            "memory_page_request",
            "entry_create",
            "entry_update",
            "entry_delete",
            "collection_trigger",
            "cursor_set",
            "cursor_clear",
            "memory_archive"
        ])
        #expect(payloads[0]["query"] == .string("recipes"))
        #expect(payloads[1]["name"] == .string("recipes"))
        #expect(payloads[2]["name"] == .string("meals"))
        #expect(payloads[2]["description"] == .string("meal notes"))
        #expect(payloads[2]["intent"] == .string("remember food"))
        #expect(payloads[2]["inclusion"] == .string("always"))
        #expect(payloads[2]["recall"] == .string("all"))
        #expect(payloads[2]["published"] == .bool(true))
        #expect(payloads[2]["collector_interval_seconds"] == .number(3_600))
        #expect(payloads[3]["section"] == .string("entries"))
        #expect(payloads[3]["offset"] == .number(1))
        #expect(payloads[4]["section"] == .string("collector_runs"))
        #expect(payloads[5]["key"] == .string("dinner"))
        #expect(payloads[5]["content"] == .string("likes soup"))
        #expect(payloads[10]["log_name"] == .string("penny.log"))
        #expect(viewModel.newName.isEmpty)
        #expect(viewModel.entryContent.isEmpty)
        client.disconnect()
    }

    @Test func settingsViewModelSavesPrefsAndUsesConfigAndPermissionSurface() async throws {
        let prefs = configuredPrefs(url: "wss://old.example/penny/", username: "alice", password: "secret")
        let (client, transport) = makeAdminClient(prefs: prefs)
        await connectAndClearStartupFrames(client, transport)
        let viewModel = SettingsViewModel(client: client, prefs: prefs)

        viewModel.refresh()
        transport.emit("""
        {
          "type": "config_response",
          "params": [{
            "key": "LLM_TEMPERATURE",
            "value": "0.7",
            "default": "0.7",
            "description": "Sampling temperature",
            "type": "float",
            "group": "llm"
          }]
        }
        """)
        transport.emit("""
        {
          "type": "domain_permissions_sync",
          "permissions": [{
            "domain": "old.example",
            "permission": "blocked"
          }]
        }
        """)
        transport.emit("""
        {
          "type": "permission_prompt",
          "request_id": "req-1",
          "domain": "new.example",
          "url": "https://new.example/article"
        }
        """)

        let param = try #require(viewModel.runtimeConfigParams.first)
        viewModel.setConfigValue(" 0.2 ", for: param)
        viewModel.saveConfigValue(for: param)
        viewModel.domainDraft = " allow.example "
        viewModel.domainPermission = .allowed
        viewModel.submitDomainPermission()
        let entry = try #require(viewModel.domainPermissions.first)
        viewModel.deleteDomain(entry)
        viewModel.decidePermissionPrompt(allowed: true)

        viewModel.webSocketURL = " wss://new.example/penny/ "
        viewModel.username = " bob "
        viewModel.password = "new-secret"
        viewModel.isMessageLayoutSwitcherEnabled = true
        viewModel.saveConnection()

        let payloads = await sentPayloads(transport, count: 5)
        #expect(Array(payloads.map(typeName).prefix(5)) == [
            "config_request",
            "config_update",
            "domain_update",
            "domain_delete",
            "permission_decision"
        ])
        #expect(payloads[1]["key"] == .string("LLM_TEMPERATURE"))
        #expect(payloads[1]["value"] == .string("0.2"))
        #expect(payloads[2]["domain"] == .string("allow.example"))
        #expect(payloads[2]["permission"] == .string("allowed"))
        #expect(payloads[3]["domain"] == .string("old.example"))
        #expect(payloads[4]["request_id"] == .string("req-1"))
        #expect(payloads[4]["allowed"] == .bool(true))
        #expect(prefs.webSocketURL == "wss://new.example/penny/")
        #expect(prefs.username == "bob")
        #expect(prefs.password == "new-secret")
        #expect(prefs.isMessageLayoutSwitcherEnabled)
        client.disconnect()
    }
}

@MainActor
private final class AdminViewModelMockTransport: WebSocketTransport {
    private(set) var sentPayloads: [[String: JSONValue]] = []
    private var onReceive: WebSocketTransport.ReceiveHandler?
    private var onFailure: WebSocketTransport.FailureHandler?
    var isConnected = false

    func connect(
        request: URLRequest,
        onReceive: @escaping WebSocketTransport.ReceiveHandler,
        onFailure: @escaping WebSocketTransport.FailureHandler
    ) {
        self.onReceive = onReceive
        self.onFailure = onFailure
        isConnected = true
    }

    func disconnect() {
        onReceive = nil
        onFailure = nil
        isConnected = false
    }

    func send(_ data: Data) async throws {
        sentPayloads.append(try JSONDecoder().decode([String: JSONValue].self, from: data))
    }

    func emit(_ json: String) {
        onReceive?(Data(json.utf8))
    }

    func clearSentPayloads() {
        sentPayloads.removeAll()
    }
}

@MainActor
private func makeAdminClient(
    prefs: Prefs? = nil
) -> (PennyWebSocketClient, AdminViewModelMockTransport) {
    let transport = AdminViewModelMockTransport()
    let client = PennyWebSocketClient(
        databaseService: configuredDatabase(),
        prefs: prefs ?? configuredPrefs(),
        webSocketClient: transport
    )
    return (client, transport)
}

@MainActor
private func connectAndClearStartupFrames(
    _ client: PennyWebSocketClient,
    _ transport: AdminViewModelMockTransport
) async {
    await client.connect()
    _ = await sentPayloads(transport, count: 2)
    transport.clearSentPayloads()
}

@MainActor
private func sentPayloads(
    _ transport: AdminViewModelMockTransport,
    count: Int
) async -> [[String: JSONValue]] {
    for _ in 0..<20 {
        if transport.sentPayloads.count >= count {
            return transport.sentPayloads
        }
        await Task.yield()
    }
    #expect(transport.sentPayloads.count >= count)
    return transport.sentPayloads
}

private func typeName(_ payload: [String: JSONValue]) -> String {
    guard case .string(let type)? = payload["type"] else { return "" }
    return type
}

private func memoryRecordJSON(name: String) -> String {
    """
    {
      "name": "\(name)",
      "type": "collection",
      "description": "Meal notes",
      "intent": "Remember cooking preferences",
      "inclusion": "relevant",
      "recall": "recent",
      "published": true,
      "archived": false,
      "extraction_prompt": "extract meals",
      "collector_interval_seconds": 3600,
      "last_collected_at": "2026-07-05T12:00:00Z",
      "entry_count": 2
    }
    """
}

private func promptLogsResponseJSON(hasMore: Bool) -> String {
    """
    {
      "type": "prompt_logs_response",
      "has_more": \(hasMore),
      "runs": [
        {
          "run_id": "run-1",
          "agent_name": "collector",
          "prompt_count": 2,
          "started_at": "2026-07-05T12:00:00Z",
          "ended_at": "2026-07-05T12:00:01Z",
          "total_duration_ms": 100,
          "total_input_tokens": 10,
          "total_output_tokens": 20,
          "run_outcome": "worked",
          "run_reason": "ok",
          "run_target": "recipes",
          "health": {
            "bailed": false,
            "no_writes": false,
            "incomplete": false,
            "tool_failures": 0,
            "degenerate_send": false,
            "flags": [],
            "regressive": false
          },
          "record": "record text",
          "prompts": []
        },
        {
          "run_id": "run-2",
          "agent_name": "collector",
          "prompt_count": 1,
          "started_at": "2026-07-05T13:00:00Z",
          "ended_at": "2026-07-05T13:00:01Z",
          "total_duration_ms": 50,
          "total_input_tokens": 5,
          "total_output_tokens": 6,
          "run_outcome": "failed",
          "run_reason": "tool failed",
          "run_target": "news",
          "health": {
            "bailed": false,
            "no_writes": false,
            "incomplete": false,
            "tool_failures": 1,
            "degenerate_send": false,
            "flags": ["tool_failures"],
            "regressive": true
          },
          "record": "failed record",
          "prompts": []
        }
      ]
    }
    """
}
