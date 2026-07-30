import Foundation

public struct RuntimeConfigParam: Decodable, Identifiable {
    public var id: String { key }
    public let key: String
    public let value: String
    public let defaultValue: String
    public let description: String
    public let type: String
    public let group: String

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

public enum RunOutcome: String, Codable {
    case failed
    case noWork = "no_work"
    case worked
    case incomplete
    case cancelled
}

public enum RunHealthFlag: String, Codable, Sendable {
    case noWorkDone = "no_work_done"
    case noWrites = "no_writes"
    case incomplete
    case toolFailures = "tool_failures"
    case halfFormedSend = "half_formed_send"
}

public struct RunHealth: Decodable, Sendable {
    public let bailed: Bool
    public let noWrites: Bool
    public let incomplete: Bool
    public let toolFailures: Int
    public let degenerateSend: Bool
    public let flags: [RunHealthFlag]
    public let regressive: Bool

    public static let empty = RunHealth(
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

public struct PromptLogRun: Decodable, Identifiable {
    public var id: String { runID }
    public let runID: String
    public let agentName: String
    public var promptCount: Int
    public let startedAt: String
    public var endedAt: String
    public var totalDurationMS: Int
    public var totalInputTokens: Int
    public var totalOutputTokens: Int
    public var runOutcome: RunOutcome?
    public var runReason: String?
    public let runTarget: String?
    public let health: RunHealth
    public let record: String
    public var prompts: [PromptLogEntry]

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

public struct PromptLogEntry: Decodable, Identifiable {
    public let id: Int
    public let timestamp: String
    public let model: String
    public let agentName: String
    public let promptType: String
    public let durationMS: Int
    public let inputTokens: Int
    public let outputTokens: Int
    public let runTarget: String?
    public let messages: [JSONValue]
    public let response: JSONValue
    public let thinking: String
    public let hasTools: Bool

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

public enum MemoryType: String, Codable {
    case collection
    case log
}

public enum MemoryInclusion: String, Codable {
    case always
    case relevant
    case never
}

public enum MemoryRecall: String, Codable {
    case all
    case relevant
    case recent
}

public enum MemorySection: String, Codable {
    case entries
    case collectorRuns = "collector_runs"
}

public struct MemoryRecord: Decodable, Identifiable {
    public var id: String { name }
    public let name: String
    public let type: MemoryType
    public let description: String
    public let intent: String?
    public let inclusion: MemoryInclusion
    public let recall: MemoryRecall
    public let published: Bool
    public let archived: Bool
    public let extractionPrompt: String?
    public let collectorIntervalSeconds: Int?
    public let lastCollectedAt: String?
    public let entryCount: Int

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

public struct MemoryEntryRecord: Decodable, Identifiable {
    public let id: Int
    public let key: String?
    public let content: String
    public let author: String
    public let createdAt: String

    private enum CodingKeys: String, CodingKey {
        case id
        case key
        case content
        case author
        case createdAt = "created_at"
    }
}

public struct CursorRecord: Decodable, Identifiable {
    public var id: String { logName }
    public let logName: String
    public let lastReadAt: String

    private enum CodingKeys: String, CodingKey {
        case logName = "log_name"
        case lastReadAt = "last_read_at"
    }
}

public struct MemoryDetail {
    public let memory: MemoryRecord
    public var entries: [MemoryEntryRecord]
    public var entriesHasMore: Bool
    public var collectorRuns: [PromptLogRun]
    public var collectorRunsHasMore: Bool
    public var cursors: [CursorRecord]

    init(payload: MemoryDetailResponsePayload) {
        memory = payload.memory
        entries = payload.entries
        entriesHasMore = payload.entriesHasMore
        collectorRuns = payload.collectorRuns
        collectorRunsHasMore = payload.collectorRunsHasMore
        cursors = payload.cursors
    }
}

public struct MemoryPage {
    public let name: String
    public let section: MemorySection
    public let entries: [MemoryEntryRecord]
    public let runs: [PromptLogRun]
    public let hasMore: Bool

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

public struct CollectionTriggerResult: Decodable {
    let name: String
    let success: Bool
    let message: String
}

public enum DomainPermission: String, Codable {
    case allowed
    case blocked
}

public struct DomainPermissionEntry: Decodable, Identifiable {
    public var id: String { domain }
    public let domain: String
    public let permission: DomainPermission
}

struct DomainPermissionsSyncPayload: Decodable {
    let permissions: [DomainPermissionEntry]
}

public struct PermissionPrompt: Decodable, Identifiable {
    public var id: String { requestID }
    public let requestID: String
    public let domain: String
    public let url: String

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

public enum JSONValue: Codable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    public init(from decoder: Decoder) throws {
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

    public func encode(to encoder: Encoder) throws {
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
