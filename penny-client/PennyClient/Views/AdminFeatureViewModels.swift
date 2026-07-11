import Foundation
import Observation

@MainActor
@Observable
final class SchedulesViewModel {
    let client: PennyWebSocketClient
    var draftCommand = ""
    var editedPromptText: [Int: String] = [:]

    init(client: PennyWebSocketClient) {
        self.client = client
    }

    var schedules: [ScheduleItem] {
        client.schedules
    }

    var errorText: String? {
        client.schedulesError
    }

    var canAddSchedule: Bool {
        !draftCommand.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func refresh() {
        client.requestSchedules()
    }

    func addSchedule() {
        let command = draftCommand.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !command.isEmpty else { return }
        draftCommand = ""
        client.addSchedule(command: command)
    }

    func promptText(for schedule: ScheduleItem) -> String {
        editedPromptText[schedule.id] ?? schedule.promptText
    }

    func setPromptText(_ promptText: String, for schedule: ScheduleItem) {
        editedPromptText[schedule.id] = promptText
    }

    func save(schedule: ScheduleItem) {
        let promptText = promptText(for: schedule).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !promptText.isEmpty else { return }
        editedPromptText[schedule.id] = promptText
        client.updateSchedule(scheduleID: schedule.id, promptText: promptText)
    }

    func delete(schedule: ScheduleItem) {
        editedPromptText[schedule.id] = nil
        client.deleteSchedule(scheduleID: schedule.id)
    }
}

@MainActor
@Observable
final class InsightsViewModel {
    let client: PennyWebSocketClient
    var selectedAgentName = ""
    var query = ""
    var flaggedOnly = false
    private(set) var requestedOffset = 0

    init(client: PennyWebSocketClient) {
        self.client = client
    }

    var runs: [PromptLogRun] {
        client.promptLogRuns
    }

    var hasMore: Bool {
        client.promptLogsHasMore
    }

    var totalPromptCount: Int {
        runs.reduce(0) { $0 + $1.promptCount }
    }

    var failedRunCount: Int {
        runs.filter { $0.runOutcome == .failed || $0.runOutcome == .incomplete }.count
    }

    func refresh() {
        requestedOffset = 0
        request(offset: nil)
    }

    func loadMore() {
        guard hasMore else { return }
        requestedOffset = runs.count
        request(offset: requestedOffset)
    }

    private func request(offset: Int?) {
        client.requestPromptLogs(
            agentName: trimmedNilIfEmpty(selectedAgentName),
            offset: offset,
            query: trimmedNilIfEmpty(query),
            flaggedOnly: flaggedOnly ? true : nil
        )
    }
}

@MainActor
@Observable
final class MemoryManagementViewModel {
    let client: PennyWebSocketClient
    var query = ""
    var selectedMemoryName: String?
    var newName = ""
    var newDescription = ""
    var newIntent = ""
    var newInclusion: MemoryInclusion = .relevant
    var newRecall: MemoryRecall = .recent
    var newPublished = false
    var newExtractionPrompt = ""
    var newCollectorIntervalText = ""
    var entryKey = ""
    var entryContent = ""

    init(client: PennyWebSocketClient) {
        self.client = client
    }

    var memories: [MemoryRecord] {
        client.memories
    }

    var detail: MemoryDetail? {
        client.memoryDetail
    }

    var page: MemoryPage? {
        client.memoryPage
    }

    var selectedMemory: MemoryRecord? {
        memories.first { $0.name == selectedMemoryName } ?? detail?.memory
    }

    var canCreateMemory: Bool {
        !newName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
            !newDescription.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
            !newIntent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var canSubmitEntry: Bool {
        selectedMemoryName != nil && !entryContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func refresh() {
        client.requestMemories(query: trimmedNilIfEmpty(query))
    }

    func select(memory: MemoryRecord) {
        selectedMemoryName = memory.name
        client.requestMemoryDetail(name: memory.name, query: trimmedNilIfEmpty(query))
    }

    func loadMoreEntries() {
        guard let name = selectedMemoryName ?? detail?.memory.name else { return }
        let offset = detail?.entries.count ?? 0
        client.requestMemoryPage(name: name, section: .entries, offset: offset, query: trimmedNilIfEmpty(query))
    }

    func loadMoreCollectorRuns() {
        guard let name = selectedMemoryName ?? detail?.memory.name else { return }
        let offset = detail?.collectorRuns.count ?? 0
        client.requestMemoryPage(name: name, section: .collectorRuns, offset: offset, query: trimmedNilIfEmpty(query))
    }

    func createMemory() {
        guard canCreateMemory else { return }

        client.createMemory(
            name: newName.trimmingCharacters(in: .whitespacesAndNewlines),
            description: newDescription.trimmingCharacters(in: .whitespacesAndNewlines),
            intent: newIntent.trimmingCharacters(in: .whitespacesAndNewlines),
            inclusion: newInclusion,
            recall: newRecall,
            published: newPublished,
            extractionPrompt: trimmedNilIfEmpty(newExtractionPrompt),
            collectorIntervalSeconds: Int(newCollectorIntervalText.trimmingCharacters(in: .whitespacesAndNewlines))
        )
        clearCreateDraft()
    }

    func updateSelectedMemory(description: String, intent: String, inclusion: MemoryInclusion, recall: MemoryRecall, published: Bool) {
        guard let name = selectedMemoryName ?? detail?.memory.name else { return }
        client.updateMemory(
            name: name,
            description: trimmedNilIfEmpty(description),
            intent: trimmedNilIfEmpty(intent),
            inclusion: inclusion,
            recall: recall,
            published: published
        )
    }

    func archiveSelectedMemory() {
        guard let name = selectedMemoryName ?? detail?.memory.name else { return }
        client.archiveMemory(name: name)
    }

    func submitEntry() {
        guard let name = selectedMemoryName ?? detail?.memory.name else { return }
        let content = entryContent.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else { return }
        client.createEntry(memory: name, key: entryKey.trimmingCharacters(in: .whitespacesAndNewlines), content: content)
        entryKey = ""
        entryContent = ""
    }

    func updateEntry(key: String?, content: String) {
        guard let name = selectedMemoryName ?? detail?.memory.name, let key, !key.isEmpty else { return }
        client.updateEntry(memory: name, key: key, content: content)
    }

    func deleteEntry(key: String?) {
        guard let name = selectedMemoryName ?? detail?.memory.name, let key, !key.isEmpty else { return }
        client.deleteEntry(memory: name, key: key)
    }

    func triggerCollection() {
        guard let name = selectedMemoryName ?? detail?.memory.name else { return }
        client.triggerCollection(name: name)
    }

    func setCursor(logName: String, lastReadAt: String) {
        guard let name = selectedMemoryName ?? detail?.memory.name else { return }
        client.setCursor(name: name, logName: logName, lastReadAt: lastReadAt)
    }

    func clearCursor(logName: String) {
        guard let name = selectedMemoryName ?? detail?.memory.name else { return }
        client.clearCursor(name: name, logName: logName)
    }

    private func clearCreateDraft() {
        newName = ""
        newDescription = ""
        newIntent = ""
        newInclusion = .relevant
        newRecall = .recent
        newPublished = false
        newExtractionPrompt = ""
        newCollectorIntervalText = ""
    }
}

@MainActor
@Observable
final class SettingsViewModel {
    let client: PennyWebSocketClient
    private let prefs: Prefs
    var webSocketURL: String
    var username: String
    var password: String
    var isMessageLayoutSwitcherEnabled: Bool
    var editedConfigValues: [String: String] = [:]
    var domainDraft = ""
    var domainPermission: DomainPermission = .allowed

    init(client: PennyWebSocketClient, prefs: Prefs) {
        self.client = client
        self.prefs = prefs
        webSocketURL = prefs.webSocketURL ?? ""
        username = prefs.username ?? ""
        password = prefs.password ?? ""
        isMessageLayoutSwitcherEnabled = prefs.isMessageLayoutSwitcherEnabled
    }

    var canSaveConnection: Bool {
        !webSocketURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var apnsHost: String {
        client.apnsHost
    }

    var runtimeConfigParams: [RuntimeConfigParam] {
        client.runtimeConfigParams
    }

    var domainPermissions: [DomainPermissionEntry] {
        client.domainPermissions
    }

    var permissionPrompt: PermissionPrompt? {
        client.permissionPrompt
    }

    var canSubmitDomain: Bool {
        !domainDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func refresh() {
        client.requestConfig()
    }

    func configValue(for param: RuntimeConfigParam) -> String {
        editedConfigValues[param.key] ?? param.value
    }

    func setConfigValue(_ value: String, for param: RuntimeConfigParam) {
        editedConfigValues[param.key] = value
    }

    func saveConfigValue(for param: RuntimeConfigParam) {
        let value = configValue(for: param).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }
        editedConfigValues[param.key] = value
        client.updateConfig(key: param.key, value: value)
    }

    func saveConnection() {
        guard canSaveConnection else { return }
        prefs.webSocketURL = webSocketURL.trimmingCharacters(in: .whitespacesAndNewlines)
        prefs.username = username.trimmingCharacters(in: .whitespacesAndNewlines)
        prefs.password = password
        prefs.isMessageLayoutSwitcherEnabled = isMessageLayoutSwitcherEnabled
        client.reconnect()
    }

    func submitDomainPermission() {
        let domain = domainDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !domain.isEmpty else { return }
        client.updateDomain(domain: domain, permission: domainPermission)
        domainDraft = ""
    }

    func deleteDomain(_ entry: DomainPermissionEntry) {
        client.deleteDomain(domain: entry.domain)
    }

    func startHistorySync(channelTypes: [String], includeAttachments: Bool) {
        client.startHistorySync(
            channelTypes: channelTypes,
            includeAttachments: includeAttachments
        )
    }

    func deleteAllMessages() {
        client.deleteAllMessages()
    }

    func decidePermissionPrompt(allowed: Bool) {
        guard let prompt = permissionPrompt else { return }
        client.decidePermission(requestID: prompt.requestID, allowed: allowed)
    }
}

private func trimmedNilIfEmpty(_ value: String) -> String? {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
}
