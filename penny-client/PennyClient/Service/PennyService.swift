import Foundation
import Observation
import SwiftUI
import UIKit
import UserNotifications

struct MessagePageCursor: Equatable, Sendable {
    let createdAt: Date
    let id: Int
}

extension MessagePageCursor: CustomStringConvertible {
    var description: String {
        "createdAt=\(createdAt.ISO8601Format()), id=\(id)"
    }
}

struct MessagePageRequest: Sendable {
    let limit: Int
    let before: MessagePageCursor?
    let filter: MessagePageFilter

    init(limit: Int = 30, before: MessagePageCursor? = nil, filter: MessagePageFilter = .all) {
        self.limit = limit
        self.before = before
        self.filter = filter
    }
}

struct MessagePage {
    let messages: [ChatMessage]
    let nextCursor: MessagePageCursor?
    let hasMore: Bool
}

enum MessagePageFilter: Equatable, Sendable {
    case all
    case penny
    case schedule
    case chat
    case notifier
    case collector

    private static let collectorPrefix = "Collector: "

    var debugDescription: String {
        switch self {
        case .all:
            return "all"
        case .penny:
            return "penny"
        case .schedule:
            return "schedule"
        case .chat:
            return "chat"
        case .notifier:
            return "notifier"
        case .collector:
            return "collector"
        }
    }

    func includes(_ message: ChatMessage) -> Bool {
        switch self {
        case .all:
            return true
        case .penny:
            return ["Penny", "Startup", "Test Push"].contains(message.sourceHint)
        case .schedule:
            return message.sourceHint == "Schedule"
        case .chat:
            return message.isOutgoing || message.sourceHint == "Chat"
        case .notifier:
            return message.sourceHint == "Notifier"
        case .collector:
            return message.sourceHint?.hasPrefix(Self.collectorPrefix) == true
        }
    }
}

@MainActor
@Observable
final class PennyService {
    private let webSocketClient: any WebSocketTransport
    private var heartbeatTask: Task<Void, Never>?
    private var notificationTokenTask: Task<Void, Never>?
    private var localMessageID = -1
    private let databaseService: DatabaseService
    private let prefs: Prefs
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    @ObservationIgnored private var liveMessages: Binding<[ChatMessage]>?
    @ObservationIgnored private var liveHasNewMessages: Binding<Bool>?
    @ObservationIgnored private var liveMessageFilter: MessagePageFilter = .all

    var messages: [ChatMessage] = []
    var pendingCount = 0
    var isConnected = false
    var isRegistered = false
    var isTyping = false
    var lastError: String?
    var runtimeConfigParams: [RuntimeConfigParam] = []
    var schedules: [ScheduleItem] = []
    var schedulesError: String?
    var promptLogRuns: [PromptLogRun] = []
    var promptLogsHasMore = false
    var memories: [MemoryRecord] = []
    var memoryDetail: MemoryDetail?
    var memoryPage: MemoryPage?
    var lastMemoryChangedName: String?
    var collectionTriggerResult: CollectionTriggerResult?
    var domainPermissions: [DomainPermissionEntry] = []
    var permissionPrompt: PermissionPrompt?

    private let pendingMessagePullLimit = 3

    init() {
        self.databaseService = .shared
        self.prefs = .shared
        self.webSocketClient = WebSocketClient()
        prepareMessageStore()
    }

    init(databaseService: DatabaseService) {
        self.databaseService = databaseService
        self.prefs = .shared
        self.webSocketClient = WebSocketClient()
        prepareMessageStore()
    }

    init(databaseService: DatabaseService, prefs: Prefs) {
        self.databaseService = databaseService
        self.prefs = prefs
        self.webSocketClient = WebSocketClient()
        prepareMessageStore()
    }

    init(databaseService: DatabaseService, prefs: Prefs, webSocketClient: any WebSocketTransport) {
        self.databaseService = databaseService
        self.prefs = prefs
        self.webSocketClient = webSocketClient
        prepareMessageStore()
    }

    private func prepareMessageStore() {
        databaseService.setup()
        localMessageID = min(-1, (databaseService.minimumMessageID() ?? 0) - 1)
    }

    var canSend: Bool {
        isConnected && isRegistered
    }

    var statusText: String {
        if let lastError {
            return lastError
        }

        if isRegistered {
            return "Connected"
        }

        if isConnected {
            return "Registering"
        }

        return "Disconnected"
    }

    var connectionColor: Color {
        if isRegistered { return .green }
        if isConnected { return .orange }
        return .red
    }

    func connect() async {
        guard !webSocketClient.isConnected else { return }

        lastError = nil
        guard let request = makeAuthenticatedRequest() else { return }

        webSocketClient.connect(
            request: request,
            onReceive: { [weak self] data in
                self?.handle(data)
            },
            onFailure: { [weak self] error in
                self?.handleReceiveFailure(error)
            }
        )

        startBackgroundTasks()
        sendRegistration()
        send(.pullMessages(limit: pendingMessagePullLimit))
    }

    func reconnect() {
        disconnect(clearLiveBindings: false)
        Task { await connect() }
    }

    func disconnect() {
        disconnect(clearLiveBindings: true)
    }

    private func disconnect(clearLiveBindings: Bool) {
        stopBackgroundTasks()
        if clearLiveBindings {
            unbindLiveMessages()
        }
        webSocketClient.disconnect()
        isConnected = false
        isRegistered = false
        isTyping = false
    }

    func requestMessagePage(_ request: MessagePageRequest) async -> MessagePage {
        debugLogFrame(
            "message page requested " +
            "(limit=\(request.limit), before=\(request.before?.description ?? "nil"), filter=\(request.filter.debugDescription))"
        )
        let page = databaseService.loadMessagePage(request)
        debugLogFrame(
            "message page returned " +
            "(count=\(page.messages.count), hasMore=\(page.hasMore), nextCursor=\(page.nextCursor?.description ?? "nil"))"
        )
        return page
    }

    func bindLiveMessages(
        _ messages: Binding<[ChatMessage]>,
        hasNewMessages: Binding<Bool>,
        filter: MessagePageFilter
    ) {
        liveMessages = messages
        liveHasNewMessages = hasNewMessages
        liveMessageFilter = filter
    }

    func updateLiveMessageFilter(_ filter: MessagePageFilter) {
        liveMessageFilter = filter
    }

    func unbindLiveMessages() {
        liveMessages = nil
        liveHasNewMessages = nil
    }

    func sendMessage(_ content: String) {
        let message = ChatMessage.local(id: nextLocalMessageID(), content: content)
        messages.append(message)
        databaseService.save(message: MessageModel(message: message))
        publishLiveMessages([message])
        send(.message(content: content))
    }

    func requestConfig() {
        send(.configRequest)
    }

    func updateConfig(key: String, value: String) {
        send(.configUpdate(key: key, value: value))
    }

    func requestSchedules() {
        send(.schedulesRequest)
    }

    func addSchedule(command: String) {
        send(.scheduleAdd(command: command))
    }

    func updateSchedule(scheduleID: Int, promptText: String) {
        send(.scheduleUpdate(scheduleID: scheduleID, promptText: promptText))
    }

    func deleteSchedule(scheduleID: Int) {
        send(.scheduleDelete(scheduleID: scheduleID))
    }

    func requestPromptLogs(
        agentName: String? = nil,
        offset: Int? = nil,
        query: String? = nil,
        flaggedOnly: Bool? = nil
    ) {
        send(.promptLogsRequest(
            agentName: agentName,
            offset: offset,
            query: query,
            flaggedOnly: flaggedOnly
        ))
    }

    func requestMemories(query: String? = nil) {
        send(.memoriesRequest(query: query))
    }

    func requestMemoryDetail(name: String, query: String? = nil) {
        send(.memoryDetailRequest(name: name, query: query))
    }

    func requestMemoryPage(
        name: String,
        section: MemorySection,
        offset: Int,
        query: String? = nil
    ) {
        send(.memoryPageRequest(name: name, section: section, offset: offset, query: query))
    }

    func createMemory(
        name: String,
        description: String,
        intent: String,
        inclusion: MemoryInclusion,
        recall: MemoryRecall,
        published: Bool? = nil,
        extractionPrompt: String? = nil,
        collectorIntervalSeconds: Int? = nil
    ) {
        send(.memoryCreate(
            name: name,
            description: description,
            intent: intent,
            inclusion: inclusion,
            recall: recall,
            published: published,
            extractionPrompt: extractionPrompt,
            collectorIntervalSeconds: collectorIntervalSeconds
        ))
    }

    func updateMemory(
        name: String,
        description: String? = nil,
        intent: String? = nil,
        inclusion: MemoryInclusion? = nil,
        recall: MemoryRecall? = nil,
        published: Bool? = nil,
        extractionPrompt: String? = nil,
        collectorIntervalSeconds: Int? = nil
    ) {
        send(.memoryUpdate(
            name: name,
            description: description,
            intent: intent,
            inclusion: inclusion,
            recall: recall,
            published: published,
            extractionPrompt: extractionPrompt,
            collectorIntervalSeconds: collectorIntervalSeconds
        ))
    }

    func archiveMemory(name: String) {
        send(.memoryArchive(name: name))
    }

    func createEntry(memory: String, key: String, content: String) {
        send(.entryCreate(memory: memory, key: key, content: content))
    }

    func updateEntry(memory: String, key: String, content: String) {
        send(.entryUpdate(memory: memory, key: key, content: content))
    }

    func deleteEntry(memory: String, key: String) {
        send(.entryDelete(memory: memory, key: key))
    }

    func triggerCollection(name: String) {
        send(.collectionTrigger(name: name))
    }

    func setCursor(name: String, logName: String, lastReadAt: String) {
        send(.cursorSet(name: name, logName: logName, lastReadAt: lastReadAt))
    }

    func clearCursor(name: String, logName: String) {
        send(.cursorClear(name: name, logName: logName))
    }

    func updateDomain(domain: String, permission: DomainPermission) {
        send(.domainUpdate(domain: domain, permission: permission))
    }

    func deleteDomain(domain: String) {
        send(.domainDelete(domain: domain))
    }

    func decidePermission(requestID: String, allowed: Bool) {
        send(.permissionDecision(requestID: requestID, allowed: allowed))
        if permissionPrompt?.requestID == requestID {
            permissionPrompt = nil
        }
    }

    func makeAuthenticatedRequest() -> URLRequest? {
        guard let path = prefs.webSocketURL, let url = URL(string: path) else {
            lastError = "Invalid WebSocket URL: \(prefs.webSocketURL ?? "none")"
            return nil
        }

        guard let username = prefs.username, let password = prefs.password else {
            lastError = "Invalid Username or Password"
            return nil
        }

        var request = URLRequest(url: url)
        let credentials = "\(username):\(password)"
        let encodedCredentials = Data(credentials.utf8).base64EncodedString()
        request.setValue("Basic \(encodedCredentials)", forHTTPHeaderField: "Authorization")
        return request
    }

    private func startBackgroundTasks() {
        stopBackgroundTasks()
        heartbeatTask = Task { [weak self] in
            await self?.heartbeatLoop()
        }
        notificationTokenTask = Task { [weak self] in
            await self?.notificationTokenLoop()
        }
    }

    private func stopBackgroundTasks() {
        heartbeatTask?.cancel()
        notificationTokenTask?.cancel()
        heartbeatTask = nil
        notificationTokenTask = nil
    }

    private func sendRegistration() {
        send(.register(RegisterPayload.current(apnsToken: PushNotificationState.shared.deviceToken)))
    }

    private func notificationTokenLoop() async {
        let updates = NotificationCenter.default.notifications(named: PushNotificationState.didUpdateDeviceToken)
        for await _ in updates {
            guard !Task.isCancelled else { return }
            sendRegistration()
        }
    }

    private func heartbeatLoop() async {
        while !Task.isCancelled {
            do {
                try await Task.sleep(for: .seconds(25))
                send(.heartbeat)
            } catch {
                return
            }
        }
    }

    private func handleReceiveFailure(_ error: Error) {
        lastError = "WebSocket receive failed: \(error.localizedDescription)"
        print(lastError ?? error.localizedDescription)
        disconnect()
    }

    private func handle(_ data: Data) {
        do {
            apply(try decoder.decode(ServerEnvelope.self, from: data))
        } catch {
            skipUndecodableMessage(data)
        }
    }

    private func skipUndecodableMessage(_ data: Data) {
        let type = (try? decoder.decode(ServerMessageType.self, from: data))?.type ?? "unknown"
        print("Skipping unrecognized server message (type: \(type))")
    }

    private func apply(_ envelope: ServerEnvelope) {
        switch envelope {
        case .status(let payload):
            isConnected = payload.connected
            lastError = payload.error
        case .registered(let payload):
            isConnected = true
            isRegistered = true
            pendingCount = payload.pendingCount
            send(.pullMessages(limit: pendingMessagePullLimit))
        case .outboxChanged(let payload):
            pendingCount = payload.pendingCount
            if payload.pendingCount > 0 {
                send(.pullMessages(limit: pendingMessagePullLimit))
            }
        case .messages(let payload):
            receive(payload.messages)
        case .messagesAcked:
            break
        case .typing(let payload):
            isTyping = payload.active
        case .configResponse(let payload):
            runtimeConfigParams = payload.params
        case .schedulesResponse(let payload):
            schedules = payload.schedules
            schedulesError = payload.error
        case .promptLogsResponse(let payload):
            if payload.runs.isEmpty || promptLogRuns.isEmpty {
                promptLogRuns = payload.runs
            } else {
                mergePromptLogRuns(payload.runs)
            }
            promptLogsHasMore = payload.hasMore
        case .promptLogUpdate(let payload):
            applyPromptLogUpdate(payload.prompt)
        case .runOutcomeUpdate(let payload):
            applyRunOutcomeUpdate(payload)
        case .memoriesResponse(let payload):
            memories = payload.memories
        case .memoryDetailResponse(let payload):
            memoryDetail = MemoryDetail(payload: payload)
        case .memoryPageResponse(let payload):
            memoryPage = MemoryPage(payload: payload)
        case .memoryChanged(let payload):
            lastMemoryChangedName = payload.name
        case .collectionTriggerResult(let payload):
            collectionTriggerResult = payload
        case .domainPermissionsSync(let payload):
            domainPermissions = payload.permissions
        case .permissionPrompt(let payload):
            permissionPrompt = payload
        case .permissionDismiss(let payload):
            if permissionPrompt?.requestID == payload.requestID {
                permissionPrompt = nil
            }
        }
    }

    private func mergePromptLogRuns(_ runs: [PromptLogRun]) {
        for run in runs {
            if let existingIndex = promptLogRuns.firstIndex(where: { $0.runID == run.runID }) {
                promptLogRuns[existingIndex] = run
            } else {
                promptLogRuns.append(run)
            }
        }
    }

    private func applyPromptLogUpdate(_ prompt: PromptLogUpdateEntry) {
        if let runIndex = promptLogRuns.firstIndex(where: { $0.runID == prompt.runID }) {
            promptLogRuns[runIndex].prompts.append(PromptLogEntry(update: prompt))
            promptLogRuns[runIndex].promptCount = promptLogRuns[runIndex].prompts.count
            promptLogRuns[runIndex].endedAt = prompt.timestamp
            promptLogRuns[runIndex].totalDurationMS += prompt.durationMS
            promptLogRuns[runIndex].totalInputTokens += prompt.inputTokens
            promptLogRuns[runIndex].totalOutputTokens += prompt.outputTokens
        } else {
            promptLogRuns.insert(PromptLogRun(update: prompt), at: 0)
        }
    }

    private func applyRunOutcomeUpdate(_ payload: RunOutcomeUpdatePayload) {
        guard let index = promptLogRuns.firstIndex(where: { $0.runID == payload.runID }) else { return }
        promptLogRuns[index].runOutcome = payload.outcome
        promptLogRuns[index].runReason = payload.reason
    }

    private func receive(_ incomingMessages: [ServerChatMessage]) {
        let incomingIDs = incomingMessages.map(\.id)
        if incomingIDs.isEmpty {
            pendingCount = 0
            return
        }

        let newMessages = incomingMessages
            .filter { !databaseService.containsMessage(serverID: $0.id) }
            .map(ChatMessage.remote)

        if !newMessages.isEmpty {
            newMessages.forEach { databaseService.save(message: MessageModel(message: $0)) }
            messages.append(contentsOf: newMessages)
            messages.sort { $0.createdAt < $1.createdAt }
            publishLiveMessages(newMessages)
        }

        send(.ackMessages(ids: incomingIDs))
        pendingCount = max(0, pendingCount - incomingIDs.count)
        clearAppBadge()

        if pendingCount > 0 {
            send(.pullMessages(limit: pendingMessagePullLimit))
        }
    }

    private func publishLiveMessages(_ newMessages: [ChatMessage]) {
        guard !newMessages.isEmpty else { return }

        let visibleMessages = newMessages.filter { liveMessageFilter.includes($0) }
        if !visibleMessages.isEmpty {
            liveMessages?.wrappedValue.append(contentsOf: visibleMessages)
        }

        if visibleMessages.count != newMessages.count {
            liveHasNewMessages?.wrappedValue = true
        }
    }

    private func clearAppBadge() {
        UNUserNotificationCenter.current().setBadgeCount(0) { error in
            if let error {
                print("Failed to clear badge count: \(error.localizedDescription)")
            }
        }
    }

    private func send(_ outgoingMessage: ClientMessage) {
        Task {
            do {
                let data = try encoder.encode(outgoingMessage)
                debugLogFrame("sent frame (\(data.count) bytes)")
                try await webSocketClient.send(data)
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                }
            }
        }
    }

    /// Logs a short, non-sensitive frame summary in debug builds only. Never logs frame
    /// contents: outbound frames carry the device secret and APNs token, inbound frames
    /// carry chat content, and on device these would land in the unified system log.
    private func debugLogFrame(_ summary: @autoclosure () -> String) {
        #if DEBUG
        print("[PennyService] \(summary())")
        #endif
    }

    private func nextLocalMessageID() -> Int {
        defer { localMessageID -= 1 }
        return localMessageID
    }
}

typealias PennyWebSocketClient = PennyService

private enum ClientMessage: Encodable {
    case register(RegisterPayload)
    case message(content: String)
    case pullMessages(limit: Int)
    case ackMessages(ids: [Int])
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
        case .ackMessages(let ids):
            try container.encode("ack_messages", forKey: .type)
            try container.encode(ids, forKey: .ids)
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
        case allowed
    }
}

private struct RegisterPayload {
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
///
/// NOTE: unverified by build on the authoring machine (no Xcode). Verify on a
/// real device / TestFlight build before relying on it.
private enum ApnsEnvironment: String {
    case sandbox
    case production

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

private struct ServerMessageType: Decodable {
    let type: String
}

private enum ServerEnvelope: Decodable {
    case status(StatusPayload)
    case registered(RegisteredPayload)
    case outboxChanged(OutboxChangedPayload)
    case messages(MessagesPayload)
    case messagesAcked(MessagesAckedPayload)
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

    private enum CodingKeys: String, CodingKey {
        case type
    }
}

private struct StatusPayload: Decodable {
    let connected: Bool
    let error: String?
}

private struct RegisteredPayload: Decodable {
    let deviceID: String
    let isDefault: Bool
    let pendingCount: Int

    private enum CodingKeys: String, CodingKey {
        case deviceID = "device_id"
        case isDefault = "is_default"
        case pendingCount = "pending_count"
    }
}

private struct OutboxChangedPayload: Decodable {
    let pendingCount: Int

    private enum CodingKeys: String, CodingKey {
        case pendingCount = "pending_count"
    }
}

private struct MessagesPayload: Decodable {
    let messages: [ServerChatMessage]
}

private struct MessagesAckedPayload: Decodable {
    let count: Int
}

private struct TypingPayload: Decodable {
    let active: Bool
}

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

private struct ConfigResponsePayload: Decodable {
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

private struct SchedulesResponsePayload: Decodable {
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

private struct PromptLogsResponsePayload: Decodable {
    let runs: [PromptLogRun]
    let hasMore: Bool

    private enum CodingKeys: String, CodingKey {
        case runs
        case hasMore = "has_more"
    }
}

private struct PromptLogUpdatePayload: Decodable {
    let prompt: PromptLogUpdateEntry
}

private struct RunOutcomeUpdatePayload: Decodable {
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

    fileprivate init(payload: MemoryDetailResponsePayload) {
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

    fileprivate init(payload: MemoryPageResponsePayload) {
        name = payload.name
        section = payload.section
        entries = payload.entries
        runs = payload.runs
        hasMore = payload.hasMore
    }
}

private struct MemoriesResponsePayload: Decodable {
    let memories: [MemoryRecord]
}

private struct MemoryDetailResponsePayload: Decodable {
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

private struct MemoryPageResponsePayload: Decodable {
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

private struct MemoryChangedPayload: Decodable {
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

private struct DomainPermissionsSyncPayload: Decodable {
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

private struct PermissionDismissPayload: Decodable {
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

private struct ServerChatMessage: Decodable {
    let id: Int
    let createdAt: Date
    let content: String
    let attachments: [Attachment]
    let sourceType: String?
    let sourceName: String?
    let sourceHint: String?
    let pushTitle: String?
    let pushSummary: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case createdAt = "created_at"
        case content
        case attachments
        case sourceType = "source_type"
        case sourceName = "source_name"
        case sourceHint = "source_hint"
        case pushTitle = "push_title"
        case pushSummary = "push_summary"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(Int.self, forKey: .id)
        content = try container.decode(String.self, forKey: .content)
        attachments = try container.decodeIfPresent([Attachment].self, forKey: .attachments) ?? []
        sourceType = try container.decodeIfPresent(String.self, forKey: .sourceType)
        sourceName = try container.decodeIfPresent(String.self, forKey: .sourceName)
        sourceHint = try container.decodeIfPresent(String.self, forKey: .sourceHint)
        pushTitle = try container.decodeIfPresent(String.self, forKey: .pushTitle)
        pushSummary = try container.decodeIfPresent(String.self, forKey: .pushSummary)

        let createdAtString = try container.decode(String.self, forKey: .createdAt)
        createdAt = DateParser.parse(createdAtString) ?? .now
    }
}

private struct Attachment: Decodable {
    let dataURL: String?
    let url: String?
    let name: String?
    let contentType: String?

    var image: UIImage? {
        guard let dataURL, let data = DataURLDecoder.decode(dataURL) else { return nil }
        return UIImage(data: data)
    }

    init(from decoder: Decoder) throws {
        if let container = try? decoder.singleValueContainer(), let dataURL = try? container.decode(String.self) {
            self.dataURL = dataURL
            url = nil
            name = nil
            contentType = nil
            return
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        dataURL = try container.decodeIfPresent(String.self, forKey: .dataURL)
        url = try container.decodeIfPresent(String.self, forKey: .url)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        contentType = try container.decodeIfPresent(String.self, forKey: .contentType)
    }

    private enum CodingKeys: String, CodingKey {
        case dataURL = "data_url"
        case url
        case name
        case contentType = "content_type"
    }
}

struct ImageAttachment: Identifiable {
    let id = UUID()
    let image: UIImage
}

struct ChatMessage: Identifiable {
    let id: Int
    let serverID: Int?
    let createdAt: Date
    let content: String
    let sourceHint: String?
    let imageAttachmentDataURLs: [String]
    let imageAttachments: [ImageAttachment]
    let isOutgoing: Bool

    var displayTime: String {
        createdAt.formatted(date: .omitted, time: .shortened)
    }

    init(
        id: Int,
        serverID: Int?,
        createdAt: Date,
        content: String,
        sourceHint: String?,
        imageAttachmentDataURLs: [String] = [],
        imageAttachments: [ImageAttachment],
        isOutgoing: Bool
    ) {
        self.id = id
        self.serverID = serverID
        self.createdAt = createdAt
        self.content = content
        self.sourceHint = sourceHint
        self.imageAttachmentDataURLs = imageAttachmentDataURLs
        self.imageAttachments = imageAttachments
        self.isOutgoing = isOutgoing
    }

    init(model: MessageModel) {
        id = model.id
        serverID = model.serverID
        createdAt = model.createdAt
        content = model.content
        sourceHint = model.sourceHint
        imageAttachmentDataURLs = model.imageAttachmentDataURLs
        imageAttachments = model.imageAttachmentDataURLs.compactMap { dataURL in
            guard let data = DataURLDecoder.decode(dataURL), let image = UIImage(data: data) else { return nil }
            return ImageAttachment(image: image)
        }
        isOutgoing = model.isOutgoing
    }

    static func local(id: Int, content: String) -> ChatMessage {
        ChatMessage(id: id, serverID: nil, createdAt: .now, content: content, sourceHint: nil, imageAttachments: [], isOutgoing: true)
    }

    fileprivate static func remote(_ message: ServerChatMessage) -> ChatMessage {
        let imageAttachmentDataURLs = message.attachments.compactMap(\.dataURL)
        let imageAttachments = message.attachments.compactMap(\.image).map(ImageAttachment.init(image:))
        return ChatMessage(id: message.id, serverID: message.id, createdAt: message.createdAt, content: message.content, sourceHint: message.sourceHint, imageAttachmentDataURLs: imageAttachmentDataURLs, imageAttachments: imageAttachments, isOutgoing: false)
    }
}

private enum DataURLDecoder {
    static func decode(_ value: String) -> Data? {
        let trimmedValue = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let commaIndex = trimmedValue.firstIndex(of: ",") else { return nil }

        let metadata = trimmedValue[..<commaIndex].lowercased()
        guard metadata.hasPrefix("data:"), metadata.contains(";base64") else { return nil }

        let base64StartIndex = trimmedValue.index(after: commaIndex)
        let base64 = trimmedValue[base64StartIndex...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\n", with: "")
            .replacingOccurrences(of: "\r", with: "")

        return Data(base64Encoded: base64)
    }
}

private enum DateParser {
    static func parse(_ value: String) -> Date? {
        if let date = iso8601WithFractionalSeconds.date(from: value) {
            return date
        }

        if let date = iso8601.date(from: value) {
            return date
        }

        if let date = localTimestampWithFractionalSeconds.date(from: value) {
            return date
        }

        return localTimestamp.date(from: value)
    }

    private static let iso8601: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private static let iso8601WithFractionalSeconds: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let localTimestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return formatter
    }()

    private static let localTimestampWithFractionalSeconds: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        return formatter
    }()
}

private enum DeviceIdentity {
    private static let keychain = SystemKeychain()
    private static let deviceIDAccount = "device_id"
    private static let deviceSecretAccount = "device_secret"

    static func stableDeviceID() -> String {
        stableUUID(account: deviceIDAccount)
    }

    static func deviceSecret() -> String {
        stableUUID(account: deviceSecretAccount)
    }

    private static func stableUUID(account: String) -> String {
        if let existingValue = keychain.string(account: account) {
            return existingValue
        }

        let newValue = UUID().uuidString
        keychain.set(newValue, account: account)
        return newValue
    }
}
