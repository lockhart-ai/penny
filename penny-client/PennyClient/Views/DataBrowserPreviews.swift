#if DEBUG
import SwiftUI
import PennyServices

/// Synthetic data shared by previews and presentation tests; never uses a live service.
@MainActor
enum DataBrowserFixtures {
    static func model(tab: DataBrowserViewModel.Tab = .memories, selection: String? = nil,
                      panel: DataBrowserViewModel.Panel = .entries, group: DataBrowserViewModel.Group = .collections,
                      expanded: Set<String> = []) -> DataBrowserViewModel {
        let model = DataBrowserViewModel(read: read)
        model.tab = tab
        model.selection = selection
        model.panel = panel
        model.group = group
        model.expanded = expanded
        return model
    }

    static func read(_ request: DataBrowserRequest) async throws -> DataBrowserResult {
        switch request {
        case .memories:
            return .memories(try [memory("price-watch"), memory("notes"), memory("messages", type: "log"), memory("archived-notes", archived: true)])
        case .detail(let name, _):
            let record = try memory(name, type: name == "messages" ? "log" : "collection", archived: name == "archived-notes")
            let entries: [JSONValue] = (1...3).map { index -> JSONValue in
                let key: JSONValue = name == "messages" ? .null : .string("Item \(index)")
                let content: String = index == 1
                    ? "Product: Example coffee beans\nLast observed price: $18\n**This is literal stored text.**"
                    : String(repeating: "Long example content remains available in full.\n", count: 24)
                let fields: [String: JSONValue] = [
                    "id": .number(Double(index)),
                    "key": key,
                    "content": .string(content),
                    "author": .string(index == 1 ? "user" : "collector"),
                    "created_at": .string("2026-09-01T12:00:00Z")
                ]
                return .object(fields)
            }
            let values: [String: JSONValue] = [
                "memory": try jsonValue(record), "entries": .array(entries), "entries_has_more": .bool(false),
                "collector_runs": .array(name == "messages" ? [] : try runs().map(jsonValue)), "collector_runs_has_more": .bool(false),
                "cursors": .array(name == "messages" ? [] : [.object(["log_name": .string("messages"), "last_read_at": .string("2026-09-01T12:00:00Z")])])
            ]
            return .detail(try JSONDecoder().decode(MemoryDetail.self, from: JSONEncoder().encode(values)))
        case .prompts: return .prompts(try runs(), hasMore: false)
        case .page: throw DataBrowserError.invalidResponse
        }
    }

    private static func memory(_ name: String, type: String = "collection", archived: Bool = false) throws -> MemoryRecord {
        let values: [String: JSONValue] = [
            "name": .string(name), "type": .string(type), "description": .string("Synthetic example information for previewing the data browser."),
            "archived": .bool(archived), "published": .bool(type == "collection"), "entry_count": .number(3),
            "schedule": type == "collection" && name != "notes" ? .string("FREQ=DAILY;BYHOUR=9") : .null,
            "extraction_prompt": type == "collection" && name != "notes" ? .string("1. log_read(memory='messages')\n2. collection_write(memory='price-watch', entries=[…])") : .null
        ]
        return try JSONDecoder().decode(MemoryRecord.self, from: JSONEncoder().encode(values))
    }

    private static func runs() throws -> [PromptLogRun] {
        try ["worked", "no_work", "failed", "incomplete", "cancelled"].map { outcome in
            let json = """
            {"run_id":"preview-\(outcome)","agent_name":"collector","prompt_count":1,"started_at":"2026-09-01T12:00:00Z",
            "ended_at":"2026-09-01T12:00:01Z","total_duration_ms":1000,"total_input_tokens":125,"total_output_tokens":30,
            "run_outcome":"\(outcome)","run_reason":"Example outcome explanation.","run_target":"price-watch","record":"Observed a price and recorded the result.",
            "health":{"bailed":false,"no_writes":false,"incomplete":false,"tool_failures":0,"degenerate_send":false,"flags":[],"regressive":false},
            "prompts":[{"id":1,"timestamp":"2026-09-01T12:00:00Z","model":"example-model","agent_name":"collector","prompt_type":"collector",
            "duration_ms":1000,"input_tokens":125,"output_tokens":30,"messages":[{"role":"user","content":"Read the latest example price."},
            {"role":"tool","content":{"product":"Coffee beans","price":18},"tool_call_id":"example-call"}],
            "response":{"tool_calls":[{"name":"collection_write","arguments":{"key":"coffee","content":"Price: $18"}}]},
            "thinking":"Compare the observation with the previous entry.","has_tools":true}]}
            """
            return try JSONDecoder().decode(PromptLogRun.self, from: Data(json.utf8))
        }
    }

    private static func jsonValue<Value: Encodable>(_ value: Value) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(value))
    }
}

private struct DataBrowserPreview: View {
    @State var model: DataBrowserViewModel
    var body: some View {
        DataBrowserContent(model: model, reconnect: {})
            .onAppear { model.updateConnection(true) }
            .onDisappear { model.stop() }
    }
}

#Preview("Memories") { DataBrowserPreview(model: DataBrowserFixtures.model()) }
#Preview("Long entries") { DataBrowserPreview(model: DataBrowserFixtures.model(selection: "price-watch")) }
#Preview("Activity") { DataBrowserPreview(model: DataBrowserFixtures.model(selection: "price-watch", panel: .activity)) }
#Preview("Read-only config") { DataBrowserPreview(model: DataBrowserFixtures.model(selection: "price-watch", panel: .config)) }
#Preview("No automation") { DataBrowserPreview(model: DataBrowserFixtures.model(selection: "notes", panel: .config)) }
#Preview("Archived") { DataBrowserPreview(model: DataBrowserFixtures.model(group: .archived)) }
#Preview("Log list") { DataBrowserPreview(model: DataBrowserFixtures.model(group: .logs)) }
#Preview("Logs") { DataBrowserPreview(model: DataBrowserFixtures.model(selection: "messages")) }
#Preview("Prompts") { DataBrowserPreview(model: DataBrowserFixtures.model(tab: .prompts)) }
#Preview("Expanded structured prompt") {
    DataBrowserPreview(model: DataBrowserFixtures.model(tab: .prompts, expanded: ["run:preview-worked", "prompt:preview-worked:1", "prompt:preview-worked:1:turn:1"]))
}
#Preview("Loading") {
    DataBrowserPreview(model: DataBrowserViewModel { _ in
        try await Task.sleep(for: .seconds(30))
        return .memories([])
    })
}
#Preview("Empty") { DataBrowserPreview(model: DataBrowserViewModel { _ in .memories([]) }) }
#Preview("Error") { DataBrowserPreview(model: DataBrowserViewModel { _ in throw DataBrowserError.serverUpdateRequired }) }
#endif
