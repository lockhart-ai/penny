import Foundation

struct RuntimeConfigParam: Decodable, Identifiable {
    var id: String { key }
    let key: String
    let value: String
    let defaultValue: String
    let description: String
    let type: String
    let group: String

    private enum CodingKeys: String, CodingKey {
        case key
        case value
        case defaultValue = "default"
        case description
        case type
        case group
    }
}

struct ConfigResponsePayload: Decodable {
    let params: [RuntimeConfigParam]
}

struct ScheduleItem: Decodable, Identifiable {
    let id: Int
    let timingDescription: String
    let promptText: String
    let cronExpression: String

    private enum CodingKeys: String, CodingKey {
        case id
        case timingDescription = "timing_description"
        case promptText = "prompt_text"
        case cronExpression = "cron_expression"
    }
}

struct SchedulesResponsePayload: Decodable {
    let schedules: [ScheduleItem]
    let error: String?
}

enum RunOutcome: String, Codable {
    case failed
    case noWork = "no_work"
    case worked
    case incomplete
    case cancelled
}

enum RunHealthFlag: String, Codable {
    case noWorkDone = "no_work_done"
    case noWrites = "no_writes"
    case incomplete
    case toolFailures = "tool_failures"
    case halfFormedSend = "half_formed_send"
}

struct RunHealth: Decodable {
    let bailed: Bool
    let noWrites: Bool
    let incomplete: Bool
    let toolFailures: Int
    let degenerateSend: Bool
    let flags: [RunHealthFlag]
    let regressive: Bool

    static let empty = RunHealth(
        bailed: false,
        noWrites: false,
        incomplete: false,
        toolFailures: 0,
        degenerateSend: false,
        flags: [],
        regressive: false
    )

    private enum CodingKeys: String, CodingKey {
        case bailed
        case noWrites = "no_writes"
        case incomplete
        case toolFailures = "tool_failures"
        case degenerateSend = "degenerate_send"
        case flags
        case regressive
    }
}

struct PromptLogRun: Decodable, Identifiable {
    var id: String { runID }
    let runID: String
    let agentName: String
    var promptCount: Int
    let startedAt: String
    var endedAt: String
    var totalDurationMS: Int
    var totalInputTokens: Int
    var totalOutputTokens: Int
    var runOutcome: RunOutcome?
    var runReason: String?
    let runTarget: String?
    let health: RunHealth
    let record: String
    var prompts: [PromptLogEntry]

    init(update: PromptLogUpdateEntry) {
        runID = update.runID
        agentName = update.agentName
        promptCount = 1
        startedAt = update.timestamp
        endedAt = update.timestamp
        totalDurationMS = update.durationMS
        totalInputTokens = update.inputTokens
        totalOutputTokens = update.outputTokens
        runOutcome = nil
        runReason = nil
        runTarget = update.runTarget
        health = .empty
        record = ""
        prompts = [PromptLogEntry(update: update)]
    }

    private enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case agentName = "agent_name"
        case promptCount = "prompt_count"
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case totalDurationMS = "total_duration_ms"
        case totalInputTokens = "total_input_tokens"
        case totalOutputTokens = "total_output_tokens"
        case runOutcome = "run_outcome"
        case runReason = "run_reason"
        case runTarget = "run_target"
        case health
        case record
        case prompts
    }
}

struct PromptLogEntry: Decodable, Identifiable {
    let id: Int
    let timestamp: String
    let model: String
    let agentName: String
    let promptType: String
    let durationMS: Int
    let inputTokens: Int
    let outputTokens: Int
    let runTarget: String?
    let messages: [JSONValue]
    let response: JSONValue
    let thinking: String
    let hasTools: Bool

    init(update: PromptLogUpdateEntry) {
        id = update.id
        timestamp = update.timestamp
        model = update.model
        agentName = update.agentName
        promptType = update.promptType
        durationMS = update.durationMS
        inputTokens = update.inputTokens
        outputTokens = update.outputTokens
        runTarget = update.runTarget
        messages = update.messages
        response = update.response
        thinking = update.thinking
        hasTools = update.hasTools
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case timestamp
        case model
        case agentName = "agent_name"
        case promptType = "prompt_type"
        case durationMS = "duration_ms"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case runTarget = "run_target"
        case messages
        case response
        case thinking
        case hasTools = "has_tools"
    }
}

struct PromptLogUpdateEntry: Decodable, Identifiable {
    let id: Int
    let runID: String
    let timestamp: String
    let model: String
    let agentName: String
    let promptType: String
    let durationMS: Int
    let inputTokens: Int
    let outputTokens: Int
    let runTarget: String?
    let messages: [JSONValue]
    let response: JSONValue
    let thinking: String
    let hasTools: Bool

    private enum CodingKeys: String, CodingKey {
        case id
        case runID = "run_id"
        case timestamp
        case model
        case agentName = "agent_name"
        case promptType = "prompt_type"
        case durationMS = "duration_ms"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case runTarget = "run_target"
        case messages
        case response
        case thinking
        case hasTools = "has_tools"
    }
}

struct PromptLogsResponsePayload: Decodable {
    let runs: [PromptLogRun]
    let hasMore: Bool

    private enum CodingKeys: String, CodingKey {
        case runs
        case hasMore = "has_more"
    }
}

struct PromptLogUpdatePayload: Decodable {
    let prompt: PromptLogUpdateEntry
}

struct RunOutcomeUpdatePayload: Decodable {
    let runID: String
    let outcome: RunOutcome
    let reason: String

    private enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case outcome
        case reason
    }
}

enum MemoryType: String, Codable {
    case collection
    case log
}

enum MemoryInclusion: String, Codable {
    case always
    case relevant
    case never
}

enum MemoryRecall: String, Codable {
    case all
    case relevant
    case recent
}

enum MemorySection: String, Codable {
    case entries
    case collectorRuns = "collector_runs"
}

struct MemoryRecord: Decodable, Identifiable {
    var id: String { name }
    let name: String
    let type: MemoryType
    let description: String
    let intent: String?
    let inclusion: MemoryInclusion
    let recall: MemoryRecall
    let published: Bool
    let archived: Bool
    let extractionPrompt: String?
    let collectorIntervalSeconds: Int?
    let lastCollectedAt: String?
    let entryCount: Int

    private enum CodingKeys: String, CodingKey {
        case name
        case type
        case description
        case intent
        case inclusion
        case recall
        case published
        case archived
        case extractionPrompt = "extraction_prompt"
        case collectorIntervalSeconds = "collector_interval_seconds"
        case lastCollectedAt = "last_collected_at"
        case entryCount = "entry_count"
    }
}

struct MemoryEntryRecord: Decodable, Identifiable {
    let id: Int
    let key: String?
    let content: String
    let author: String
    let createdAt: String

    private enum CodingKeys: String, CodingKey {
        case id
        case key
        case content
        case author
        case createdAt = "created_at"
    }
}

struct CursorRecord: Decodable, Identifiable {
    var id: String { logName }
    let logName: String
    let lastReadAt: String

    private enum CodingKeys: String, CodingKey {
        case logName = "log_name"
        case lastReadAt = "last_read_at"
    }
}

struct MemoryDetail {
    let memory: MemoryRecord
    var entries: [MemoryEntryRecord]
    var entriesHasMore: Bool
    var collectorRuns: [PromptLogRun]
    var collectorRunsHasMore: Bool
    var cursors: [CursorRecord]

    init(payload: MemoryDetailResponsePayload) {
        memory = payload.memory
        entries = payload.entries
        entriesHasMore = payload.entriesHasMore
        collectorRuns = payload.collectorRuns
        collectorRunsHasMore = payload.collectorRunsHasMore
        cursors = payload.cursors
    }
}

struct MemoryPage {
    let name: String
    let section: MemorySection
    let entries: [MemoryEntryRecord]
    let runs: [PromptLogRun]
    let hasMore: Bool

    init(payload: MemoryPageResponsePayload) {
        name = payload.name
        section = payload.section
        entries = payload.entries
        runs = payload.runs
        hasMore = payload.hasMore
    }
}

struct MemoriesResponsePayload: Decodable {
    let memories: [MemoryRecord]
}

struct MemoryDetailResponsePayload: Decodable {
    let memory: MemoryRecord
    let entries: [MemoryEntryRecord]
    let entriesHasMore: Bool
    let collectorRuns: [PromptLogRun]
    let collectorRunsHasMore: Bool
    let cursors: [CursorRecord]

    private enum CodingKeys: String, CodingKey {
        case memory
        case entries
        case entriesHasMore = "entries_has_more"
        case collectorRuns = "collector_runs"
        case collectorRunsHasMore = "collector_runs_has_more"
        case cursors
    }
}

struct MemoryPageResponsePayload: Decodable {
    let name: String
    let section: MemorySection
    let entries: [MemoryEntryRecord]
    let runs: [PromptLogRun]
    let hasMore: Bool

    private enum CodingKeys: String, CodingKey {
        case name
        case section
        case entries
        case runs
        case hasMore = "has_more"
    }
}

struct MemoryChangedPayload: Decodable {
    let name: String?
}

struct CollectionTriggerResult: Decodable {
    let name: String
    let success: Bool
    let message: String
}

enum DomainPermission: String, Codable {
    case allowed
    case blocked
}

struct DomainPermissionEntry: Decodable, Identifiable {
    var id: String { domain }
    let domain: String
    let permission: DomainPermission
}

struct DomainPermissionsSyncPayload: Decodable {
    let permissions: [DomainPermissionEntry]
}

struct PermissionPrompt: Decodable, Identifiable {
    var id: String { requestID }
    let requestID: String
    let domain: String
    let url: String

    private enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case domain
        case url
    }
}

struct PermissionDismissPayload: Decodable {
    let requestID: String

    private enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
    }
}

enum JSONValue: Codable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}
