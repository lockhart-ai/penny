import Foundation
import Testing
@testable import PennyClient

@Suite(.serialized)
@MainActor
struct PennyServiceAdminSurfaceTests {
    @Test func configRequestsEncodeExpectedFrames() async throws {
        let (service, transport) = makeSubject()

        service.requestConfig()
        service.updateConfig(key: "LLM_TEMPERATURE", value: "0.2")

        let payloads = await sentPayloads(transport, count: 2)
        #expect(payloads[0]["type"] == .string("config_request"))
        #expect(payloads[1]["type"] == .string("config_update"))
        #expect(payloads[1]["key"] == .string("LLM_TEMPERATURE"))
        #expect(payloads[1]["value"] == .string("0.2"))
    }

    @Test func scheduleRequestsEncodeExpectedFrames() async throws {
        let (service, transport) = makeSubject()

        service.requestSchedules()
        service.addSchedule(command: "summarize unread mail")
        service.updateSchedule(scheduleID: 42, promptText: "new prompt")
        service.deleteSchedule(scheduleID: 42)

        let payloads = await sentPayloads(transport, count: 4)
        #expect(payloads[0]["type"] == .string("schedules_request"))
        #expect(payloads[1]["type"] == .string("schedule_add"))
        #expect(payloads[1]["command"] == .string("summarize unread mail"))
        #expect(payloads[2]["type"] == .string("schedule_update"))
        #expect(payloads[2]["schedule_id"] == .number(42))
        #expect(payloads[2]["prompt_text"] == .string("new prompt"))
        #expect(payloads[3]["type"] == .string("schedule_delete"))
        #expect(payloads[3]["schedule_id"] == .number(42))
    }

    @Test func promptLogRequestEncodesFilters() async throws {
        let (service, transport) = makeSubject()

        service.requestPromptLogs(agentName: "collector", offset: 50, query: "failed", flaggedOnly: true)

        let payloads = await sentPayloads(transport, count: 1)
        let payload = try #require(payloads.first)
        #expect(payload["type"] == .string("prompt_logs_request"))
        #expect(payload["agent_name"] == .string("collector"))
        #expect(payload["offset"] == .number(50))
        #expect(payload["query"] == .string("failed"))
        #expect(payload["flagged_only"] == .bool(true))
    }

    @Test func memoryRequestsAndMutationsEncodeExpectedFrames() async throws {
        let (service, transport) = makeSubject()

        service.requestMemories(query: "recipes")
        service.requestMemoryDetail(name: "recipes", query: "pasta")
        service.requestMemoryPage(name: "recipes", section: .collectorRuns, offset: 25, query: "failed")
        service.createMemory(
            name: "recipes",
            description: "Meal notes",
            intent: "Remember cooking preferences",
            inclusion: .relevant,
            recall: .recent,
            published: true,
            extractionPrompt: "extract meals",
            collectorIntervalSeconds: 3_600
        )
        service.updateMemory(
            name: "recipes",
            description: "Updated meals",
            inclusion: .always,
            recall: .all,
            published: false,
            extractionPrompt: "extract preferences",
            collectorIntervalSeconds: 7_200
        )
        service.archiveMemory(name: "recipes")
        service.createEntry(memory: "recipes", key: "pasta", content: "likes cacio e pepe")
        service.updateEntry(memory: "recipes", key: "pasta", content: "likes carbonara")
        service.deleteEntry(memory: "recipes", key: "pasta")
        service.triggerCollection(name: "recipes")
        service.setCursor(name: "recipes", logName: "penny.log", lastReadAt: "2026-07-05T12:00:00Z")
        service.clearCursor(name: "recipes", logName: "penny.log")

        let payloads = await sentPayloads(transport, count: 12)
        #expect(payloads.map(typeName) == [
            "memories_request",
            "memory_detail_request",
            "memory_page_request",
            "memory_create",
            "memory_update",
            "memory_archive",
            "entry_create",
            "entry_update",
            "entry_delete",
            "collection_trigger",
            "cursor_set",
            "cursor_clear"
        ])

        #expect(payloads[2]["section"] == .string("collector_runs"))
        #expect(payloads[2]["offset"] == .number(25))
        #expect(payloads[3]["inclusion"] == .string("relevant"))
        #expect(payloads[3]["recall"] == .string("recent"))
        #expect(payloads[3]["collector_interval_seconds"] == .number(3_600))
        #expect(payloads[4]["published"] == .bool(false))
        #expect(payloads[6]["content"] == .string("likes cacio e pepe"))
        #expect(payloads[10]["log_name"] == .string("penny.log"))
        #expect(payloads[10]["last_read_at"] == .string("2026-07-05T12:00:00Z"))
    }

    @Test func domainAndPermissionActionsEncodeExpectedFrames() async throws {
        let (service, transport) = makeSubject()

        service.updateDomain(domain: "example.com", permission: .allowed)
        service.deleteDomain(domain: "blocked.test")
        service.decidePermission(requestID: "req-1", allowed: false)

        let payloads = await sentPayloads(transport, count: 3)
        #expect(payloads[0]["type"] == .string("domain_update"))
        #expect(payloads[0]["domain"] == .string("example.com"))
        #expect(payloads[0]["permission"] == .string("allowed"))
        #expect(payloads[1]["type"] == .string("domain_delete"))
        #expect(payloads[1]["domain"] == .string("blocked.test"))
        #expect(payloads[2]["type"] == .string("permission_decision"))
        #expect(payloads[2]["request_id"] == .string("req-1"))
        #expect(payloads[2]["allowed"] == .bool(false))
    }

    @Test func appliesConfigScheduleAndDomainResponses() async throws {
        let (service, transport) = makeSubject()
        await connectAndClearStartupFrames(service, transport)

        transport.emit("""
        {
          "type": "config_response",
          "params": [{
            "key": "LLM_TEMPERATURE",
            "value": "0.2",
            "default": "0.7",
            "description": "Sampling temperature",
            "type": "float",
            "group": "llm"
          }]
        }
        """)
        transport.emit("""
        {
          "type": "schedules_response",
          "error": null,
          "schedules": [{
            "id": 9,
            "timing_description": "every hour",
            "prompt_text": "summarize",
            "cron_expression": "0 * * * *"
          }]
        }
        """)
        transport.emit("""
        {
          "type": "domain_permissions_sync",
          "permissions": [{
            "domain": "example.com",
            "permission": "blocked"
          }]
        }
        """)

        #expect(service.runtimeConfigParams.first?.key == "LLM_TEMPERATURE")
        #expect(service.runtimeConfigParams.first?.defaultValue == "0.7")
        #expect(service.schedules.first?.id == 9)
        #expect(service.schedules.first?.promptText == "summarize")
        #expect(service.schedulesError == nil)
        #expect(service.domainPermissions.first?.domain == "example.com")
        #expect(service.domainPermissions.first?.permission == .blocked)
        service.disconnect()
    }

    @Test func appliesPromptLogResponsesAndUpdates() async throws {
        let (service, transport) = makeSubject()
        await connectAndClearStartupFrames(service, transport)

        transport.emit("""
        {
          "type": "prompt_logs_response",
          "has_more": true,
          "runs": [{
            "run_id": "run-1",
            "agent_name": "collector",
            "prompt_count": 1,
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
            "prompts": [{
              "id": 1,
              "run_id": "run-1",
              "timestamp": "2026-07-05T12:00:00Z",
              "model": "gpt-test",
              "agent_name": "collector",
              "prompt_type": "collect",
              "duration_ms": 100,
              "input_tokens": 10,
              "output_tokens": 20,
              "run_target": "recipes",
              "messages": [],
              "response": {"ok": true},
              "thinking": "",
              "has_tools": true
            }]
          }]
        }
        """)
        transport.emit("""
        {
          "type": "prompt_log_update",
          "prompt": {
            "id": 2,
            "run_id": "run-1",
            "timestamp": "2026-07-05T12:00:02Z",
            "model": "gpt-test",
            "agent_name": "collector",
            "prompt_type": "collect",
            "duration_ms": 50,
            "input_tokens": 3,
            "output_tokens": 4,
            "run_target": "recipes",
            "messages": [],
            "response": "done",
            "thinking": "brief",
            "has_tools": false
          }
        }
        """)
        transport.emit("""
        {
          "type": "run_outcome_update",
          "run_id": "run-1",
          "outcome": "incomplete",
          "reason": "needs review"
        }
        """)

        let run = try #require(service.promptLogRuns.first)
        #expect(service.promptLogsHasMore)
        #expect(run.runID == "run-1")
        #expect(run.promptCount == 2)
        #expect(run.prompts.map(\.id) == [1, 2])
        #expect(run.totalDurationMS == 150)
        #expect(run.totalInputTokens == 13)
        #expect(run.totalOutputTokens == 24)
        #expect(run.runOutcome == .incomplete)
        #expect(run.runReason == "needs review")
        service.disconnect()
    }

    @Test func appliesMemoryAndCollectionResponses() async throws {
        let (service, transport) = makeSubject()
        await connectAndClearStartupFrames(service, transport)

        transport.emit("""
        {
          "type": "memories_response",
          "memories": [\(memoryRecordJSON(name: "recipes"))]
        }
        """)
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
          "entries_has_more": false,
          "collector_runs": [],
          "collector_runs_has_more": true,
          "cursors": [{
            "log_name": "penny.log",
            "last_read_at": "2026-07-05T12:00:00Z"
          }]
        }
        """)
        transport.emit("""
        {
          "type": "memory_page_response",
          "name": "recipes",
          "section": "entries",
          "entries": [{
            "id": 8,
            "key": "dessert",
            "content": "likes tiramisu",
            "author": "assistant",
            "created_at": "2026-07-05T12:01:00Z"
          }],
          "runs": [],
          "has_more": true
        }
        """)
        transport.emit("""
        {
          "type": "memory_changed",
          "name": "recipes"
        }
        """)
        transport.emit("""
        {
          "type": "collection_trigger_result",
          "name": "recipes",
          "success": true,
          "message": "Collection queued"
        }
        """)

        #expect(service.memories.first?.name == "recipes")
        #expect(service.memoryDetail?.entries.first?.key == "pasta")
        #expect(service.memoryDetail?.collectorRunsHasMore == true)
        #expect(service.memoryDetail?.cursors.first?.logName == "penny.log")
        #expect(service.memoryPage?.section == .entries)
        #expect(service.memoryPage?.entries.first?.key == "dessert")
        #expect(service.memoryPage?.hasMore == true)
        #expect(service.lastMemoryChangedName == "recipes")
        #expect(service.collectionTriggerResult?.success == true)
        #expect(service.collectionTriggerResult?.message == "Collection queued")
        service.disconnect()
    }

    @Test func permissionPromptAndDismissUpdateState() async throws {
        let (service, transport) = makeSubject()
        await connectAndClearStartupFrames(service, transport)

        transport.emit("""
        {
          "type": "permission_prompt",
          "request_id": "req-1",
          "domain": "example.com",
          "url": "https://example.com/article"
        }
        """)

        #expect(service.permissionPrompt?.requestID == "req-1")
        #expect(service.permissionPrompt?.domain == "example.com")

        transport.emit("""
        {
          "type": "permission_dismiss",
          "request_id": "req-1"
        }
        """)

        #expect(service.permissionPrompt == nil)
        service.disconnect()
    }
}

@MainActor
private final class MockWebSocketTransport: WebSocketTransport {
    private(set) var connectedRequest: URLRequest?
    private(set) var sentPayloads: [[String: JSONValue]] = []
    private(set) var sentStrings: [String] = []
    private var onReceive: WebSocketTransport.ReceiveHandler?
    private var onFailure: WebSocketTransport.FailureHandler?
    var isConnected = false

    func connect(
        request: URLRequest,
        onReceive: @escaping WebSocketTransport.ReceiveHandler,
        onFailure: @escaping WebSocketTransport.FailureHandler
    ) {
        connectedRequest = request
        self.onReceive = onReceive
        self.onFailure = onFailure
        isConnected = true
    }

    func disconnect() {
        connectedRequest = nil
        onReceive = nil
        onFailure = nil
        isConnected = false
    }

    func send(_ data: Data) async throws {
        if let string = String(bytes: data, encoding: .utf8) {
            sentStrings.append(string)
        }
        sentPayloads.append(try JSONDecoder().decode([String: JSONValue].self, from: data))
    }

    func emit(_ json: String) {
        onReceive?(Data(json.utf8))
    }

    func clearSentPayloads() {
        sentPayloads.removeAll()
        sentStrings.removeAll()
    }
}

@MainActor
private func makeSubject() -> (PennyService, MockWebSocketTransport) {
    let transport = MockWebSocketTransport()
    let service = PennyService(
        databaseService: configuredDatabase(),
        prefs: configuredPrefs(),
        webSocketClient: transport
    )
    return (service, transport)
}

@MainActor
private func connectAndClearStartupFrames(
    _ service: PennyService,
    _ transport: MockWebSocketTransport
) async {
    await service.connect()
    _ = await sentPayloads(transport, count: 2)
    transport.clearSentPayloads()
}

@MainActor
private func sentPayloads(
    _ transport: MockWebSocketTransport,
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
