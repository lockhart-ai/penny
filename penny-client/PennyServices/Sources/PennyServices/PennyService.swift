import Foundation
import Observation
import SwiftUI
import UIKit
import UserNotifications

public struct AgentProgressToolItem: Identifiable {
    public let id = UUID()
    public let name: String
    public let arguments: [String: AgentProgressValue]
}

public struct AgentProgressStepItem: Identifiable {
    public let id = UUID()
    public let number: Int
    public let maxSteps: Int?
    public var tools: [AgentProgressToolItem] = []
}

public struct AgentProgressRunItem: Identifiable {
    public let id: String
    public let agent: String
    public let scope: AgentProgressScope
    public var steps: [AgentProgressStepItem] = []
}

@MainActor
@Observable
public final class PennyService {
    private let webSocketClient: any WebSocketTransport
    private var heartbeatTask: Task<Void, Never>?
    private var notificationTokenTask: Task<Void, Never>?
    private var localMessageID = -1
    private let databaseService: DatabaseService
    private let prefs: Prefs
    private let logger = OSLogService(category: .pennyService)
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    @ObservationIgnored var liveMessages: Binding<[ChatMessage]>?
    @ObservationIgnored private var liveHasNewMessages: Binding<Bool>?
    @ObservationIgnored private var historicalMessagesHandler: (([ChatMessage]) -> Void)?
    @ObservationIgnored private var liveMessageFilter: MessagePageFilter = .all
    @ObservationIgnored private var historySyncTask: Task<Void, Never>?
    @ObservationIgnored private var historyResponseContinuation: AsyncStream<HistorySyncEvent>.Continuation?
    @ObservationIgnored private var historyPageIsNewest = false
    @ObservationIgnored private var embeddingContinuations: [String: CheckedContinuation<Data, Error>] = [:]
    @ObservationIgnored private var pendingResponseTimings: [String: [PendingResponseTiming]] = [:]

    public var messages: [ChatMessage] = []
    public var pendingCount = 0
    public var isConnected = false
    public var isRegistered = false
    public var isTyping = false
    public var agentProgressRuns: [String: AgentProgressRunItem] = [:]

    public var foregroundProgress: AgentProgressRunItem? {
        agentProgressRuns.values.first(where: { $0.scope == .foreground })
    }

    public var backgroundProgressRuns: [AgentProgressRunItem] {
        agentProgressRuns.values.filter { $0.scope == .background }.sorted { $0.id < $1.id }
    }
    public var lastError: String?
    public var runtimeConfigParams: [RuntimeConfigParam] = []
    public var promptLogRuns: [PromptLogRun] = []
    public var promptLogsHasMore = false
    public var memories: [MemoryRecord] = []
    public var memoryDetail: MemoryDetail?
    public var memoryPage: MemoryPage?
    public var lastMemoryChangedName: String?
    var collectionTriggerResult: CollectionTriggerResult?
    public var domainPermissions: [DomainPermissionEntry] = []
    public var permissionPrompt: PermissionPrompt?
    public var historySyncing = false
    var historyRequestedCount = 0
    var historySavedOrUpdatedCount = 0
    var historyRemainingCount = 0
    public var historyStatus = "Not started"

    public var historyProgressText: String {
        "Requested \(historyRequestedCount) · Saved \(historySavedOrUpdatedCount) · Remaining \(historyRemainingCount)"
    }

    private let pendingMessagePullLimit = 3

    public init() {
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

    public var canSend: Bool {
        isConnected && isRegistered
    }

    public var apnsHost: String {
        ApnsEnvironment.current.host
    }

    public var statusText: String {
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

    public var connectionColor: Color {
        if isRegistered { return .green }
        if isConnected { return .orange }
        return .red
    }
}
extension PennyService {
    public func connect() async {
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

    public func reconnect() {
        disconnect(clearLiveBindings: false)
        Task { await connect() }
    }

    public func disconnect() {
        disconnect(clearLiveBindings: true)
    }

    private func disconnect(clearLiveBindings: Bool) {
        stopBackgroundTasks()
        historyResponseContinuation?.finish()
        historyResponseContinuation = nil
        historySyncTask?.cancel()
        historySyncTask = nil
        let pendingEmbeddingRequests = Array(embeddingContinuations.values)
        embeddingContinuations.removeAll()
        pendingResponseTimings.removeAll()
        for continuation in pendingEmbeddingRequests {
            continuation.resume(throwing: PennyEmbeddingError.disconnected)
        }
        if clearLiveBindings {
            unbindLiveMessages()
        }
        webSocketClient.disconnect()
        isConnected = false
        isRegistered = false
        isTyping = false
        agentProgressRuns.removeAll()
    }

    public func requestMessagePage(_ request: MessagePageRequest) async -> MessagePage {
        logger.debug(
            "message page requested " +
            "(limit=\(request.limit), before=\(request.before?.description ?? "nil"), filter=\(request.filter.debugDescription))",
            privacy: .public
        )
        let page = databaseService.loadMessagePage(request)
        logger.debug(
            "message page returned " +
            "(count=\(page.messages.count), hasMore=\(page.hasMore), nextCursor=\(page.nextCursor?.description ?? "nil"))",
            privacy: .public
        )
        return page
    }

    func requestEmbedding(_ text: String) async throws -> Data {
        guard canSend else { throw PennyEmbeddingError.disconnected }
        let requestID = UUID().uuidString
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                embeddingContinuations[requestID] = continuation
                send(.embeddingRequest(requestID: requestID, text: text))
            }
        } onCancel: {
            Task { @MainActor [weak self] in
                self?.cancelEmbeddingRequest(requestID)
            }
        }
    }

    private func cancelEmbeddingRequest(_ requestID: String) {
        guard let continuation = embeddingContinuations.removeValue(forKey: requestID) else { return }
        continuation.resume(throwing: CancellationError())
    }

    public func bindLiveMessages(
        _ messages: Binding<[ChatMessage]>,
        hasNewMessages: Binding<Bool>,
        filter: MessagePageFilter,
        historicalMessages: (([ChatMessage]) -> Void)? = nil
    ) {
        liveMessages = messages
        liveHasNewMessages = hasNewMessages
        liveMessageFilter = filter
        historicalMessagesHandler = historicalMessages
    }

    public func updateLiveMessageFilter(_ filter: MessagePageFilter) {
        liveMessageFilter = filter
    }

    public func startHistorySync(channelTypes: [String], includeAttachments: Bool = true) {
        guard !channelTypes.isEmpty else {
            historyStatus = "Select at least one channel"
            return
        }
        guard !historySyncing else { return }

        historySyncTask?.cancel()
        let state = (prefs.value(HistorySyncState.self, forKey: .historySyncState)
            .flatMap { existing in
                existing.channelTypes == channelTypes && existing.includeAttachments == includeAttachments
                    ? existing
                    : nil
            }) ?? HistorySyncState(
                channelTypes: channelTypes,
                includeAttachments: includeAttachments,
                cursor: nil,
                requestedCount: 0,
                savedOrUpdatedCount: 0,
                remainingCount: 0,
                totalCount: nil
            )
        saveHistorySyncState(state)
        applyHistoryProgress(state, status: "Starting...")
        historySyncing = true
        historySyncTask = makeHistorySyncTask(state: state)
    }

    private func makeHistorySyncTask(state: HistorySyncState) -> Task<Void, Never> {
        Task { [weak self] in
            await self?.crawlHistory(state: state)
        }
    }

    private func saveHistorySyncState(_ state: HistorySyncState?) {
        prefs.set(state, forKey: .historySyncState)
    }

    private func applyHistoryProgress(_ state: HistorySyncState, status: String? = nil) {
        historyRequestedCount = state.requestedCount
        historySavedOrUpdatedCount = state.savedOrUpdatedCount
        historyRemainingCount = state.remainingCount
        if let status { historyStatus = status }
    }

    public func deleteAllMessages() {
        historyResponseContinuation?.finish()
        historyResponseContinuation = nil
        historySyncTask?.cancel()
        historySyncTask = nil
        historySyncing = false
        historyRequestedCount = 0
        historySavedOrUpdatedCount = 0
        historyRemainingCount = 0
        historyStatus = "Not started"
        saveHistorySyncState(nil)
        databaseService.deleteAllMessages()
        messages.removeAll()
        liveMessages?.wrappedValue = []
        liveHasNewMessages?.wrappedValue = false
    }

    public func unbindLiveMessages() {
        liveMessages = nil
        liveHasNewMessages = nil
        historicalMessagesHandler = nil
    }

    public func sendMessage(_ content: String) {
        let message = ChatMessage.local(id: nextLocalMessageID(), content: content)
        messages.append(message)
        databaseService.save(message: MessageModel(message: message))
        publishLiveMessages([message])
        send(.message(content: content))
    }

    public func requestConfig() {
        send(.configRequest)
    }

    public func updateConfig(key: String, value: String) {
        send(.configUpdate(key: key, value: value))
    }

    public func requestPromptLogs(
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

    public func requestMemories(query: String? = nil) {
        send(.memoriesRequest(query: query))
    }

    public func requestMemoryDetail(name: String, query: String? = nil) {
        send(.memoryDetailRequest(name: name, query: query))
    }

    public func requestMemoryPage(
        name: String,
        section: MemorySection,
        offset: Int,
        query: String? = nil
    ) {
        send(.memoryPageRequest(name: name, section: section, offset: offset, query: query))
    }

    public func createMemory(
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

    public func updateMemory(
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

    public func archiveMemory(name: String) {
        send(.memoryArchive(name: name))
    }

    public func createEntry(memory: String, key: String, content: String) {
        send(.entryCreate(memory: memory, key: key, content: content))
    }

    public func updateEntry(memory: String, key: String, content: String) {
        send(.entryUpdate(memory: memory, key: key, content: content))
    }

    public func deleteEntry(memory: String, key: String) {
        send(.entryDelete(memory: memory, key: key))
    }

    public func triggerCollection(name: String) {
        send(.collectionTrigger(name: name))
    }

    public func setCursor(name: String, logName: String, lastReadAt: String) {
        send(.cursorSet(name: name, logName: logName, lastReadAt: lastReadAt))
    }

    public func clearCursor(name: String, logName: String) {
        send(.cursorClear(name: name, logName: logName))
    }

    public func updateDomain(domain: String, permission: DomainPermission) {
        send(.domainUpdate(domain: domain, permission: permission))
    }

    public func deleteDomain(domain: String) {
        send(.domainDelete(domain: domain))
    }

    public func decidePermission(requestID: String, allowed: Bool) {
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
}

extension PennyService {
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
        logger.error("WebSocket receive failed: \(error.localizedDescription)", privacy: .public)
        disconnect()
    }

    private func handle(_ data: Data) {
        do {
            let envelope = try decoder.decode(ServerEnvelope.self, from: data)
            apply(envelope)
        } catch {
            skipUndecodableMessage(data)
        }
    }

    private func skipUndecodableMessage(_ data: Data) {
        let type = (try? decoder.decode(ServerMessageType.self, from: data))?.type ?? "unknown"
        logger.warning("Skipping unrecognized server message (type: \(type))", privacy: .public)
    }

    private func apply(_ envelope: ServerEnvelope) {
        logResponseLatency(for: envelope)

        switch envelope {
        case .status(let payload):
            isConnected = payload.connected
            lastError = payload.error
            if historySyncing, let error = payload.error {
                historyResponseContinuation?.yield(.error(error))
            }
        case .registered(let payload):
            isConnected = true
            isRegistered = true
            pendingCount = payload.pendingCount
            send(.pullMessages(limit: pendingMessagePullLimit))
            resumeHistorySyncIfNeeded()
        case .outboxChanged(let payload):
            pendingCount = payload.pendingCount
            if payload.pendingCount > 0 {
                send(.pullMessages(limit: pendingMessagePullLimit))
            }
        case .messages(let payload):
            if payload.mode == "history_count" {
                historyResponseContinuation?.yield(.count(payload.totalCount ?? 0))
                break
            }
            let receiveResult = receive(payload)
            let newMessages = receiveResult.newMessages
            if payload.mode == "history" {
                if historyPageIsNewest {
                    let visibleMessages = newMessages.filter { liveMessageFilter.includes($0) }
                    if let historicalMessagesHandler {
                        historicalMessagesHandler(visibleMessages)
                    } else {
                        publishLiveMessages(visibleMessages)
                    }
                }
                historyResponseContinuation?.yield(.page(HistoryPageResult(
                    payload: payload,
                    savedOrUpdatedCount: receiveResult.savedOrUpdatedCount
                )))
            }
        case .messagesAcked:
            break
        case .embeddingResponse(let payload):
            guard let continuation = embeddingContinuations.removeValue(forKey: payload.requestID) else {
                break
            }
            if let error = payload.error {
                continuation.resume(throwing: PennyEmbeddingError.unavailable(error))
            } else if let encoded = payload.embedding, let data = Data(base64Encoded: encoded) {
                continuation.resume(returning: data)
            } else {
                continuation.resume(throwing: PennyEmbeddingError.invalidResponse)
            }
        case .typing(let payload):
            isTyping = payload.active
        case .agentProgress(let payload):
            applyAgentProgress(payload)
        case .configResponse(let payload):
            runtimeConfigParams = payload.params
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

    private func applyAgentProgress(_ payload: AgentProgressPayload) {
        switch payload.event {
        case .runStarted:
            guard agentProgressRuns[payload.runID] == nil else { return }
            agentProgressRuns[payload.runID] = AgentProgressRunItem(
                id: payload.runID,
                agent: payload.agent,
                scope: payload.scope
            )
        case .stepStarted:
            guard var run = agentProgressRuns[payload.runID], let step = payload.step else { return }
            guard !run.steps.contains(where: { $0.number == step }) else { return }
            run.steps.append(AgentProgressStepItem(number: step, maxSteps: payload.maxSteps))
            agentProgressRuns[payload.runID] = run
        case .toolsStarted:
            guard var run = agentProgressRuns[payload.runID] else { return }
            guard let step = payload.step ?? run.steps.last?.number,
                  let index = run.steps.firstIndex(where: { $0.number == step }) else { return }
            run.steps[index].tools.append(contentsOf: payload.tools.map {
                AgentProgressToolItem(name: $0.name, arguments: $0.arguments)
            })
            agentProgressRuns[payload.runID] = run
        case .runFinished:
            agentProgressRuns.removeValue(forKey: payload.runID)
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

    private struct ReceiveResult {
        let newMessages: [ChatMessage]
        let savedOrUpdatedCount: Int
    }

    private func receive(_ payload: MessagesPayload) -> ReceiveResult {
        // Deduplicate repeated canonical IDs before updating persistence or UI.
        var seenMessageIDs = Set<Int>()
        let incomingMessages = payload.messages.filter { seenMessageIDs.insert($0.canonicalID).inserted }
        let incomingIDs = incomingMessages.map(\.id)
        if incomingIDs.isEmpty {
            if payload.mode == "outbox" {
                pendingCount = 0
            }
            return ReceiveResult(newMessages: [], savedOrUpdatedCount: 0)
        }

        incomingMessages.forEach { message in
            guard let messageID = message.messageID,
                  let outboxID = message.outboxID else { return }
            databaseService.reconcileLegacyMessage(
                outboxID: outboxID,
                canonicalID: messageID
            )
        }

        let decodedMessages = incomingMessages.map(ChatMessage.remote)
        for message in decodedMessages where message.isOutgoing {
            guard let localID = databaseService.reconcileLocalMessage(
                content: message.content,
                createdAt: message.createdAt,
                canonicalID: message.id
            ) else { continue }
            replaceOptimisticMessage(localID: localID, with: message)
        }
        let newMessages = decodedMessages.filter {
            !databaseService.containsMessage(serverID: $0.id)
        }

        // Persist the canonical representation, including metadata corrections.
        let savedOrUpdatedCount = databaseService.saveMessages(
            decodedMessages.map(MessageModel.init(message:)),
            preserveAttachments: payload.mode == "history" && !payload.attachmentsIncluded
        )

        if !newMessages.isEmpty {
            messages.append(contentsOf: newMessages)
            messages.sort { $0.createdAt < $1.createdAt }
        }

        if payload.mode == "outbox" {
            publishLiveMessages(decodedMessages)
            send(.ackMessages(ids: incomingIDs))
            pendingCount = max(0, pendingCount - incomingIDs.count)
            clearAppBadge()
        }

        if pendingCount > 0 {
            send(.pullMessages(limit: pendingMessagePullLimit))
        }
        return ReceiveResult(newMessages: newMessages, savedOrUpdatedCount: savedOrUpdatedCount)
    }
}

extension PennyService {
    private func resumeHistorySyncIfNeeded() {
        guard !historySyncing, let state: HistorySyncState = prefs.value(
            HistorySyncState.self,
            forKey: .historySyncState
        ) else { return }
        applyHistoryProgress(state, status: "Resuming...")
        historySyncing = true
        historySyncTask = makeHistorySyncTask(state: state)
    }

    private func crawlHistory(state initialState: HistorySyncState) async {
        let stream = AsyncStream<HistorySyncEvent> { continuation in
            historyResponseContinuation = continuation
        }
        var iterator = stream.makeAsyncIterator()
        var state = initialState

        defer {
            historyResponseContinuation?.finish()
            historyResponseContinuation = nil
            historySyncTask = nil
            if historySyncing {
                historySyncing = false
            }
        }

        while !Task.isCancelled {
            guard isConnected, isRegistered else {
                historyStatus = "Waiting for connection"
                return
            }

            let countOnly = state.totalCount == nil
            historyPageIsNewest = !countOnly && state.cursor == nil && state.requestedCount == 0
            send(.historyRequest(
                limit: 30,
                before: countOnly ? nil : state.cursor,
                channelTypes: state.channelTypes,
                includeAttachments: state.includeAttachments,
                countOnly: countOnly
            ))
            guard let event = await iterator.next() else { return }
            if case .count(let totalCount) = event {
                state.totalCount = totalCount
                state.remainingCount = totalCount
                saveHistorySyncState(state)
                applyHistoryProgress(state)
                continue
            }
            guard case .page(let result) = event else {
                if case .error(let error) = event {
                    historyStatus = "Sync failed: \(error)"
                }
                return
            }

            state.requestedCount += result.payload.messages.count
            state.savedOrUpdatedCount += result.savedOrUpdatedCount
            state.remainingCount = max((state.totalCount ?? state.requestedCount) - state.requestedCount, 0)
            state.cursor = result.payload.nextCursor
            saveHistorySyncState(state)
            applyHistoryProgress(state)
            historyStatus = "In Progress"
            guard result.payload.hasMore, let nextCursor = result.payload.nextCursor else {
                saveHistorySyncState(nil)
                historyStatus = "Complete"
                return
            }
            state.cursor = nextCursor
            saveHistorySyncState(state)

            do {
                try await Task.sleep(for: .milliseconds(10))
            } catch {
                return
            }
        }
    }

    private func publishLiveMessages(_ newMessages: [ChatMessage]) {
        guard !newMessages.isEmpty else { return }

        let visibleMessages = newMessages.filter { liveMessageFilter.includes($0) }
        if !visibleMessages.isEmpty {
            var mergedMessages = liveMessages?.wrappedValue ?? []
            for message in visibleMessages where !mergedMessages.contains(where: { $0.id == message.id }) {
                mergedMessages.append(message)
            }
            mergedMessages.sort {
                $0.createdAt < $1.createdAt || ($0.createdAt == $1.createdAt && $0.id < $1.id)
            }
            liveMessages?.wrappedValue = mergedMessages
        }

        if visibleMessages.count != newMessages.count {
            liveHasNewMessages?.wrappedValue = true
        }
    }

    private func send(_ outgoingMessage: ClientMessage) {
        Task {
            do {
                let data = try encoder.encode(outgoingMessage)
                let pendingResponse = recordPendingResponse(for: outgoingMessage)
                do {
                    try await webSocketClient.send(data)
                } catch {
                    removePendingResponse(pendingResponse)
                    throw error
                }
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                }
            }
        }
    }

    private struct PendingResponseTiming {
        let id = UUID()
        let responseKey: String
        let startDate = Date()
    }

    private func recordPendingResponse(for outgoingMessage: ClientMessage) -> PendingResponseTiming? {
        guard let responseKey = outgoingMessage.expectedResponseLogKey else { return nil }
        let timing = PendingResponseTiming(responseKey: responseKey)
        pendingResponseTimings[responseKey, default: []].append(timing)
        return timing
    }

    private func removePendingResponse(_ timing: PendingResponseTiming?) {
        guard let timing,
              var timings = pendingResponseTimings[timing.responseKey] else { return }

        timings.removeAll { $0.id == timing.id }
        if timings.isEmpty {
            pendingResponseTimings[timing.responseKey] = nil
        } else {
            pendingResponseTimings[timing.responseKey] = timings
        }
    }

    private func logResponseLatency(for envelope: ServerEnvelope) {
        guard let responseKey = envelope.responseLogKey,
              var timings = pendingResponseTimings[responseKey],
              !timings.isEmpty else { return }

        let timing = timings.removeFirst()
        if timings.isEmpty {
            pendingResponseTimings[responseKey] = nil
        } else {
            pendingResponseTimings[responseKey] = timings
        }

        let elapsedMS = Int(Date().timeIntervalSince(timing.startDate) * 1_000)
        logger.debug(
            "websocket response latency (type=\(envelope.logType), elapsed_ms=\(elapsedMS))",
            privacy: .public
        )
    }

    private func nextLocalMessageID() -> Int {
        defer { localMessageID -= 1 }
        return localMessageID
    }
}
