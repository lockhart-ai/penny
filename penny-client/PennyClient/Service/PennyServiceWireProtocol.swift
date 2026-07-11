import Foundation
import UIKit

typealias PennyWebSocketClient = PennyService

enum ClientMessage: Encodable {
    case register(RegisterPayload)
    case message(content: String)
    case pullMessages(limit: Int)
    case historyRequest(limit: Int, before: String?, channelTypes: [String]?, includeAttachments: Bool, countOnly: Bool)
    case ackMessages(ids: [Int])
    case embeddingRequest(requestID: String, text: String)
    case heartbeat
    case configRequest
    case configUpdate(key: String, value: String)
    case schedulesRequest
    case scheduleAdd(command: String)
    case scheduleUpdate(scheduleID: Int, promptText: String)
    case scheduleDelete(scheduleID: Int)
    case promptLogsRequest(agentName: String?, offset: Int?, query: String?, flaggedOnly: Bool?)
    case memoriesRequest(query: String?)
    case memoryDetailRequest(name: String, query: String?)
    case memoryPageRequest(name: String, section: MemorySection, offset: Int, query: String?)
    case memoryCreate(
        name: String,
        description: String,
        intent: String,
        inclusion: MemoryInclusion,
        recall: MemoryRecall,
        published: Bool?,
        extractionPrompt: String?,
        collectorIntervalSeconds: Int?
    )
    case memoryUpdate(
        name: String,
        description: String?,
        intent: String?,
        inclusion: MemoryInclusion?,
        recall: MemoryRecall?,
        published: Bool?,
        extractionPrompt: String?,
        collectorIntervalSeconds: Int?
    )
    case memoryArchive(name: String)
    case entryCreate(memory: String, key: String, content: String)
    case entryUpdate(memory: String, key: String, content: String)
    case entryDelete(memory: String, key: String)
    case collectionTrigger(name: String)
    case cursorSet(name: String, logName: String, lastReadAt: String)
    case cursorClear(name: String, logName: String)
    case domainUpdate(domain: String, permission: DomainPermission)
    case domainDelete(domain: String)
        case permissionDecision(requestID: String, allowed: Bool)

    // swiftlint:disable:next function_body_length
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .register(let payload):
            try container.encode("register", forKey: .type)
            try container.encode(payload.deviceID, forKey: .deviceID)
            try container.encode(payload.label, forKey: .label)
            try container.encodeIfPresent(payload.pairingToken, forKey: .pairingToken)
            try container.encodeIfPresent(payload.deviceSecret, forKey: .deviceSecret)
            try container.encodeIfPresent(payload.apnsToken, forKey: .apnsToken)
            try container.encode(payload.apnsEnvironment, forKey: .apnsEnvironment)
            try container.encode(payload.appVersion, forKey: .appVersion)
        case .message(let content):
            try container.encode("message", forKey: .type)
            try container.encode(content, forKey: .content)
        case .pullMessages(let limit):
            try container.encode("pull_messages", forKey: .type)
            try container.encode(limit, forKey: .limit)
        case .historyRequest(let limit, let before, let channelTypes, let includeAttachments, let countOnly):
            try container.encode("history_request", forKey: .type)
            try container.encode(limit, forKey: .limit)
            try container.encodeIfPresent(before, forKey: .before)
            try container.encodeIfPresent(channelTypes, forKey: .channelTypes)
            try container.encode(includeAttachments, forKey: .includeAttachments)
            try container.encode(countOnly, forKey: .countOnly)
        case .ackMessages(let ids):
            try container.encode("ack_messages", forKey: .type)
            try container.encode(ids, forKey: .ids)
        case .embeddingRequest(let requestID, let text):
            try container.encode("embedding_request", forKey: .type)
            try container.encode(requestID, forKey: .requestID)
            try container.encode(text, forKey: .text)
        case .heartbeat:
            try container.encode("heartbeat", forKey: .type)
        case .configRequest:
            try container.encode("config_request", forKey: .type)
        case .configUpdate(let key, let value):
            try container.encode("config_update", forKey: .type)
            try container.encode(key, forKey: .key)
            try container.encode(value, forKey: .value)
        case .schedulesRequest:
            try container.encode("schedules_request", forKey: .type)
        case .scheduleAdd(let command):
            try container.encode("schedule_add", forKey: .type)
            try container.encode(command, forKey: .command)
        case .scheduleUpdate(let scheduleID, let promptText):
            try container.encode("schedule_update", forKey: .type)
            try container.encode(scheduleID, forKey: .scheduleID)
            try container.encode(promptText, forKey: .promptText)
        case .scheduleDelete(let scheduleID):
            try container.encode("schedule_delete", forKey: .type)
            try container.encode(scheduleID, forKey: .scheduleID)
        case .promptLogsRequest(let agentName, let offset, let query, let flaggedOnly):
            try container.encode("prompt_logs_request", forKey: .type)
            try container.encodeIfPresent(agentName, forKey: .agentName)
            try container.encodeIfPresent(offset, forKey: .offset)
            try container.encodeIfPresent(query, forKey: .query)
            try container.encodeIfPresent(flaggedOnly, forKey: .flaggedOnly)
        case .memoriesRequest(let query):
            try container.encode("memories_request", forKey: .type)
            try container.encodeIfPresent(query, forKey: .query)
        case .memoryDetailRequest(let name, let query):
            try container.encode("memory_detail_request", forKey: .type)
            try container.encode(name, forKey: .name)
            try container.encodeIfPresent(query, forKey: .query)
        case .memoryPageRequest(let name, let section, let offset, let query):
            try container.encode("memory_page_request", forKey: .type)
            try container.encode(name, forKey: .name)
            try container.encode(section, forKey: .section)
            try container.encode(offset, forKey: .offset)
            try container.encodeIfPresent(query, forKey: .query)
        case .memoryCreate(
            let name,
            let description,
            let intent,
            let inclusion,
            let recall,
            let published,
            let extractionPrompt,
            let collectorIntervalSeconds
        ):
            try container.encode("memory_create", forKey: .type)
            try container.encode(name, forKey: .name)
            try container.encode(description, forKey: .description)
            try container.encode(intent, forKey: .intent)
            try container.encode(inclusion, forKey: .inclusion)
            try container.encode(recall, forKey: .recall)
            try container.encodeIfPresent(published, forKey: .published)
            try container.encodeIfPresent(extractionPrompt, forKey: .extractionPrompt)
            try container.encodeIfPresent(collectorIntervalSeconds, forKey: .collectorIntervalSeconds)
        case .memoryUpdate(
            let name,
            let description,
            let intent,
            let inclusion,
            let recall,
            let published,
            let extractionPrompt,
            let collectorIntervalSeconds
        ):
            try container.encode("memory_update", forKey: .type)
            try container.encode(name, forKey: .name)
            try container.encodeIfPresent(description, forKey: .description)
            try container.encodeIfPresent(intent, forKey: .intent)
            try container.encodeIfPresent(inclusion, forKey: .inclusion)
            try container.encodeIfPresent(recall, forKey: .recall)
            try container.encodeIfPresent(published, forKey: .published)
            try container.encodeIfPresent(extractionPrompt, forKey: .extractionPrompt)
            try container.encodeIfPresent(collectorIntervalSeconds, forKey: .collectorIntervalSeconds)
        case .memoryArchive(let name):
            try container.encode("memory_archive", forKey: .type)
            try container.encode(name, forKey: .name)
        case .entryCreate(let memory, let key, let content):
            try container.encode("entry_create", forKey: .type)
            try container.encode(memory, forKey: .memory)
            try container.encode(key, forKey: .key)
            try container.encode(content, forKey: .content)
        case .entryUpdate(let memory, let key, let content):
            try container.encode("entry_update", forKey: .type)
            try container.encode(memory, forKey: .memory)
            try container.encode(key, forKey: .key)
            try container.encode(content, forKey: .content)
        case .entryDelete(let memory, let key):
            try container.encode("entry_delete", forKey: .type)
            try container.encode(memory, forKey: .memory)
            try container.encode(key, forKey: .key)
        case .collectionTrigger(let name):
            try container.encode("collection_trigger", forKey: .type)
            try container.encode(name, forKey: .name)
        case .cursorSet(let name, let logName, let lastReadAt):
            try container.encode("cursor_set", forKey: .type)
            try container.encode(name, forKey: .name)
            try container.encode(logName, forKey: .logName)
            try container.encode(lastReadAt, forKey: .lastReadAt)
        case .cursorClear(let name, let logName):
            try container.encode("cursor_clear", forKey: .type)
            try container.encode(name, forKey: .name)
            try container.encode(logName, forKey: .logName)
        case .domainUpdate(let domain, let permission):
            try container.encode("domain_update", forKey: .type)
            try container.encode(domain, forKey: .domain)
            try container.encode(permission, forKey: .permission)
        case .domainDelete(let domain):
            try container.encode("domain_delete", forKey: .type)
            try container.encode(domain, forKey: .domain)
        case .permissionDecision(let requestID, let allowed):
            try container.encode("permission_decision", forKey: .type)
            try container.encode(requestID, forKey: .requestID)
            try container.encode(allowed, forKey: .allowed)
        }
    }

    var expectedResponseLogKey: String? {
        switch self {
        case .register:
            return "registered"
        case .message:
            return "messages:outbox"
        case .pullMessages:
            return "messages:outbox"
        case .historyRequest:
            return "messages:history"
        case .ackMessages:
            return "messages_acked"
        case .embeddingRequest(let requestID, _):
            return "embedding_response:\(requestID)"
        case .heartbeat:
            return nil
        case .configRequest, .configUpdate:
            return "config_response"
        case .schedulesRequest, .scheduleAdd, .scheduleUpdate, .scheduleDelete:
            return "schedules_response"
        case .promptLogsRequest:
            return "prompt_logs_response"
        case .memoriesRequest:
            return "memories_response"
        case .memoryDetailRequest:
            return "memory_detail_response"
        case .memoryPageRequest:
            return "memory_page_response"
        case .memoryCreate, .memoryUpdate, .memoryArchive, .entryCreate, .entryUpdate, .entryDelete, .cursorSet, .cursorClear:
            return "memory_changed"
        case .collectionTrigger:
            return "collection_trigger_result"
        case .domainUpdate, .domainDelete:
            return "domain_permissions_sync"
        case .permissionDecision:
            return "permission_dismiss"
        }
    }

    var logType: String {
        switch self {
        case .register:
            return "register"
        case .message:
            return "message"
        case .pullMessages:
            return "pull_messages"
        case .historyRequest:
            return "history_request"
        case .ackMessages:
            return "ack_messages"
        case .embeddingRequest:
            return "embedding_request"
        case .heartbeat:
            return "heartbeat"
        case .configRequest:
            return "config_request"
        case .configUpdate:
            return "config_update"
        case .schedulesRequest:
            return "schedules_request"
        case .scheduleAdd:
            return "schedule_add"
        case .scheduleUpdate:
            return "schedule_update"
        case .scheduleDelete:
            return "schedule_delete"
        case .promptLogsRequest:
            return "prompt_logs_request"
        case .memoriesRequest:
            return "memories_request"
        case .memoryDetailRequest:
            return "memory_detail_request"
        case .memoryPageRequest:
            return "memory_page_request"
        case .memoryCreate:
            return "memory_create"
        case .memoryUpdate:
            return "memory_update"
        case .memoryArchive:
            return "memory_archive"
        case .entryCreate:
            return "entry_create"
        case .entryUpdate:
            return "entry_update"
        case .entryDelete:
            return "entry_delete"
        case .collectionTrigger:
            return "collection_trigger"
        case .cursorSet:
            return "cursor_set"
        case .cursorClear:
            return "cursor_clear"
        case .domainUpdate:
            return "domain_update"
        case .domainDelete:
            return "domain_delete"
        case .permissionDecision:
            return "permission_decision"
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case deviceID = "device_id"
        case label
        case pairingToken = "pairing_token"
        case deviceSecret = "device_secret"
        case apnsToken = "apns_token"
        case apnsEnvironment = "apns_environment"
        case appVersion = "app_version"
        case content
        case limit
        case before
        case channelTypes = "channel_types"
        case includeAttachments = "include_attachments"
        case countOnly = "count_only"
        case ids
        case key
        case value
        case command
        case scheduleID = "schedule_id"
        case promptText = "prompt_text"
        case agentName = "agent_name"
        case offset
        case query
        case flaggedOnly = "flagged_only"
        case name
        case section
        case description
        case intent
        case inclusion
        case recall
        case published
        case extractionPrompt = "extraction_prompt"
        case collectorIntervalSeconds = "collector_interval_seconds"
        case memory
        case logName = "log_name"
        case lastReadAt = "last_read_at"
        case domain
        case permission
        case requestID = "request_id"
        case text
        case allowed
    }
}

struct RegisterPayload {
    let deviceID: String
    let label: String
    let pairingToken: String?
    let deviceSecret: String?
    let apnsToken: String?
    let apnsEnvironment: String
    let appVersion: String

    static func current(apnsToken: String?) -> RegisterPayload {
        RegisterPayload(
            deviceID: DeviceIdentity.stableDeviceID(),
            label: UIDevice.current.name,
            pairingToken: "pairing-token",
            deviceSecret: DeviceIdentity.deviceSecret(),
            apnsToken: apnsToken,
            apnsEnvironment: ApnsEnvironment.current.rawValue,
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0.0"
        )
    }
}

/// APNs environment this build's push token was minted for.
///
/// A token is only valid against the matching APNs host, so the server must be
/// told which one to use. Historically hardcoded to `sandbox`, which silently
/// broke push on TestFlight/App Store builds (those carry a production token).
///
/// Derivation: DEBUG builds always use the development (sandbox) environment.
/// Release builds read the `aps-environment` entitlement from the embedded
/// provisioning profile — `development` (a dev or ad-hoc signed build, or a
/// direct device install) maps to sandbox; `production` — as in a TestFlight or
/// App Store build, or the absence of an embedded profile — maps to production.

enum ApnsEnvironment: String {
    case sandbox
    case production

    var host: String {
        switch self {
        case .sandbox:
            return "api.sandbox.push.apple.com"
        case .production:
            return "api.push.apple.com"
        }
    }

    static var current: ApnsEnvironment {
        #if DEBUG
        return .sandbox
        #else
        return embeddedProfileEnvironment() ?? .production
        #endif
    }

    private static func embeddedProfileEnvironment() -> ApnsEnvironment? {
        guard
            let url = Bundle.main.url(forResource: "embedded", withExtension: "mobileprovision"),
            let data = try? Data(contentsOf: url),
            let entitlements = provisioningEntitlements(from: data),
            let apsEnvironment = entitlements["aps-environment"] as? String
        else {
            return nil
        }
        return apsEnvironment == "development" ? .sandbox : .production
    }

    private static func provisioningEntitlements(from data: Data) -> [String: Any]? {
        // A .mobileprovision is a CMS (PKCS#7) blob wrapping an XML plist; slice
        // out the plist between the <plist ...> and </plist> markers and parse it.
        guard
            let start = data.range(of: Data("<plist".utf8))?.lowerBound,
            let end = data.range(of: Data("</plist>".utf8))?.upperBound
        else {
            return nil
        }
        let plistData = data.subdata(in: start..<end)
        let profile = try? PropertyListSerialization.propertyList(from: plistData, options: [], format: nil)
        return (profile as? [String: Any])?["Entitlements"] as? [String: Any]
    }
}

struct ServerMessageType: Decodable {
    let type: String
}

enum ServerEnvelope: Decodable {
    case status(StatusPayload)
    case registered(RegisteredPayload)
    case outboxChanged(OutboxChangedPayload)
    case messages(MessagesPayload)
    case messagesAcked(MessagesAckedPayload)
    case embeddingResponse(EmbeddingResponsePayload)
    case typing(TypingPayload)
    case configResponse(ConfigResponsePayload)
    case schedulesResponse(SchedulesResponsePayload)
    case promptLogsResponse(PromptLogsResponsePayload)
    case promptLogUpdate(PromptLogUpdatePayload)
    case runOutcomeUpdate(RunOutcomeUpdatePayload)
    case memoriesResponse(MemoriesResponsePayload)
    case memoryDetailResponse(MemoryDetailResponsePayload)
    case memoryPageResponse(MemoryPageResponsePayload)
    case memoryChanged(MemoryChangedPayload)
    case collectionTriggerResult(CollectionTriggerResult)
    case domainPermissionsSync(DomainPermissionsSyncPayload)
    case permissionPrompt(PermissionPrompt)
    case permissionDismiss(PermissionDismissPayload)

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "status":
            self = .status(try StatusPayload(from: decoder))
        case "registered":
            self = .registered(try RegisteredPayload(from: decoder))
        case "outbox_changed":
            self = .outboxChanged(try OutboxChangedPayload(from: decoder))
        case "messages":
            self = .messages(try MessagesPayload(from: decoder))
        case "messages_acked":
            self = .messagesAcked(try MessagesAckedPayload(from: decoder))
        case "embedding_response":
            self = .embeddingResponse(try EmbeddingResponsePayload(from: decoder))
        case "typing":
            self = .typing(try TypingPayload(from: decoder))
        case "config_response":
            self = .configResponse(try ConfigResponsePayload(from: decoder))
        case "schedules_response":
            self = .schedulesResponse(try SchedulesResponsePayload(from: decoder))
        case "prompt_logs_response":
            self = .promptLogsResponse(try PromptLogsResponsePayload(from: decoder))
        case "prompt_log_update":
            self = .promptLogUpdate(try PromptLogUpdatePayload(from: decoder))
        case "run_outcome_update":
            self = .runOutcomeUpdate(try RunOutcomeUpdatePayload(from: decoder))
        case "memories_response":
            self = .memoriesResponse(try MemoriesResponsePayload(from: decoder))
        case "memory_detail_response":
            self = .memoryDetailResponse(try MemoryDetailResponsePayload(from: decoder))
        case "memory_page_response":
            self = .memoryPageResponse(try MemoryPageResponsePayload(from: decoder))
        case "memory_changed":
            self = .memoryChanged(try MemoryChangedPayload(from: decoder))
        case "collection_trigger_result":
            self = .collectionTriggerResult(try CollectionTriggerResult(from: decoder))
        case "domain_permissions_sync":
            self = .domainPermissionsSync(try DomainPermissionsSyncPayload(from: decoder))
        case "permission_prompt":
            self = .permissionPrompt(try PermissionPrompt(from: decoder))
        case "permission_dismiss":
            self = .permissionDismiss(try PermissionDismissPayload(from: decoder))
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "Unknown server message type: \(type)")
        }
    }

    var responseLogKey: String? {
        switch self {
        case .status:
            return nil
        case .registered:
            return "registered"
        case .outboxChanged:
            return nil
        case .messages(let payload):
            return "messages:\(payload.mode)"
        case .messagesAcked:
            return "messages_acked"
        case .embeddingResponse(let payload):
            return "embedding_response:\(payload.requestID)"
        case .typing:
            return nil
        case .configResponse:
            return "config_response"
        case .schedulesResponse:
            return "schedules_response"
        case .promptLogsResponse:
            return "prompt_logs_response"
        case .promptLogUpdate:
            return nil
        case .runOutcomeUpdate:
            return nil
        case .memoriesResponse:
            return "memories_response"
        case .memoryDetailResponse:
            return "memory_detail_response"
        case .memoryPageResponse:
            return "memory_page_response"
        case .memoryChanged:
            return "memory_changed"
        case .collectionTriggerResult:
            return "collection_trigger_result"
        case .domainPermissionsSync:
            return "domain_permissions_sync"
        case .permissionPrompt:
            return nil
        case .permissionDismiss:
            return "permission_dismiss"
        }
    }

    var logType: String {
        switch self {
        case .status:
            return "status"
        case .registered:
            return "registered"
        case .outboxChanged:
            return "outbox_changed"
        case .messages:
            return "messages"
        case .messagesAcked:
            return "messages_acked"
        case .embeddingResponse:
            return "embedding_response"
        case .typing:
            return "typing"
        case .configResponse:
            return "config_response"
        case .schedulesResponse:
            return "schedules_response"
        case .promptLogsResponse:
            return "prompt_logs_response"
        case .promptLogUpdate:
            return "prompt_log_update"
        case .runOutcomeUpdate:
            return "run_outcome_update"
        case .memoriesResponse:
            return "memories_response"
        case .memoryDetailResponse:
            return "memory_detail_response"
        case .memoryPageResponse:
            return "memory_page_response"
        case .memoryChanged:
            return "memory_changed"
        case .collectionTriggerResult:
            return "collection_trigger_result"
        case .domainPermissionsSync:
            return "domain_permissions_sync"
        case .permissionPrompt:
            return "permission_prompt"
        case .permissionDismiss:
            return "permission_dismiss"
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type
    }
}

struct StatusPayload: Decodable {
    let connected: Bool
    let error: String?
}

struct RegisteredPayload: Decodable {
    let deviceID: String
    let isDefault: Bool
    let pendingCount: Int

    private enum CodingKeys: String, CodingKey {
        case deviceID = "device_id"
        case isDefault = "is_default"
        case pendingCount = "pending_count"
    }
}

struct OutboxChangedPayload: Decodable {
    let pendingCount: Int

    private enum CodingKeys: String, CodingKey {
        case pendingCount = "pending_count"
    }
}

struct MessagesPayload: Decodable {
    let messages: [ServerChatMessage]
    let mode: String
    let nextCursor: String?
    let hasMore: Bool
    let totalCount: Int?
    let attachmentsIncluded: Bool

    private enum CodingKeys: String, CodingKey {
        case messages
        case mode
        case nextCursor = "next_cursor"
        case hasMore = "has_more"
        case totalCount = "total_count"
        case attachmentsIncluded = "attachments_included"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        messages = try container.decode([ServerChatMessage].self, forKey: .messages)
        mode = try container.decodeIfPresent(String.self, forKey: .mode) ?? "outbox"
        nextCursor = try container.decodeIfPresent(String.self, forKey: .nextCursor)
        hasMore = try container.decodeIfPresent(Bool.self, forKey: .hasMore) ?? false
        totalCount = try container.decodeIfPresent(Int.self, forKey: .totalCount)
        attachmentsIncluded = try container.decodeIfPresent(Bool.self, forKey: .attachmentsIncluded) ?? true
    }
}

struct MessagesAckedPayload: Decodable {
    let count: Int
}

struct EmbeddingResponsePayload: Decodable {
    let requestID: String
    let embedding: String?
    let error: String?

    private enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case embedding
        case error
    }
}

struct TypingPayload: Decodable {
    let active: Bool
}
