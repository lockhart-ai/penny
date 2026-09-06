import Foundation

public enum DataBrowserRequest: Sendable {
    case memories(query: String?)
    case detail(name: String, query: String?)
    case page(name: String, section: MemorySection, offset: Int, query: String?)
    case prompts(agent: String?, query: String?, flagged: Bool, offset: Int)

    var message: ClientMessage {
        switch self {
        case .memories(let query): return .memoriesRequest(query: query)
        case .detail(let name, let query): return .memoryDetailRequest(name: name, query: query)
        case .page(let name, let section, let offset, let query):
            return .memoryPageRequest(name: name, section: section, offset: offset, query: query)
        case .prompts(let agent, let query, let flagged, let offset):
            return .promptLogsRequest(agentName: agent, offset: offset, query: query, flaggedOnly: flagged)
        }
    }
}

public enum DataBrowserResult: Sendable {
    case memories([MemoryRecord])
    case detail(MemoryDetail)
    case page(MemoryPage)
    case prompts([PromptLogRun], hasMore: Bool)
}

public enum DataBrowserError: LocalizedError {
    case disconnected
    case timedOut
    case serverUpdateRequired
    case invalidResponse
    case server(String)

    public var errorDescription: String? {
        switch self {
        case .disconnected: return "Disconnected. Reconnect to browse Penny data."
        case .timedOut: return "The data request timed out. Try again."
        case .serverUpdateRequired: return "Update the Penny server to support the data browser, then retry."
        case .invalidResponse: return "Penny returned an invalid data browser response."
        case .server(let message): return message
        }
    }
}

/// A presentation owns its tasks; this transport owns their correlated continuations.
@MainActor
public final class DataBrowserClient {
    private struct Pending {
        let responseType: String?
        let continuation: CheckedContinuation<DataBrowserResult, Error>
        let startedAt: ContinuousClock.Instant
        let timeout: Task<Void, Never>
    }

    private let send: @MainActor (Data) async throws -> Void
    private let timeout: Duration
    private var pending: [String: Pending] = [:]
    private let logger = OSLogService(category: .pennyService)

    init(timeout: Duration = .seconds(30), send: @escaping @MainActor (Data) async throws -> Void) {
        self.timeout = timeout
        self.send = send
    }

    public func read(_ request: DataBrowserRequest) async throws -> DataBrowserResult {
        let requestID = UUID().uuidString
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                begin(request, id: requestID, continuation: continuation)
            }
        } onCancel: {
            Task { @MainActor in self.finish(requestID, result: .failure(CancellationError())) }
        }
    }

    private func begin(_ request: DataBrowserRequest, id: String, continuation: CheckedContinuation<DataBrowserResult, Error>) {
        let timeoutTask = Task { [weak self, timeout] in
            do {
                try await Task.sleep(for: timeout)
                self?.finish(id, result: .failure(DataBrowserError.timedOut))
            } catch is CancellationError {
                // A response or cancellation already completed the request.
            } catch {
                self?.finish(id, result: .failure(error))
            }
        }
        pending[id] = Pending(responseType: request.message.expectedResponseLogKey, continuation: continuation,
                              startedAt: .now, timeout: timeoutTask)
        Task {
            do {
                guard pending[id] != nil else { return }
                try await send(JSONEncoder().encode(CorrelatedRead(request: request, requestID: id)))
            } catch {
                finish(id, result: .failure(error))
            }
        }
    }

    func disconnect() {
        for id in Array(pending.keys) { finish(id, result: .failure(DataBrowserError.disconnected)) }
    }

    /// Consume correlated responses before the legacy observable admin snapshots.
    func receive(_ data: Data) -> Bool {
        guard let header = try? JSONDecoder().decode(ReadHeader.self, from: data) else { return false }
        guard let requestID = header.requestID else {
            let matching = pending.filter { $0.value.responseType == header.type }.map(\.key)
            for id in matching { finish(id, result: .failure(DataBrowserError.serverUpdateRequired)) }
            return !matching.isEmpty
        }
        guard let operation = pending[requestID] else { return header.isReadResponse }
        guard header.type == operation.responseType || header.type == "data_read_error" else {
            finish(requestID, result: .failure(DataBrowserError.invalidResponse))
            return true
        }
        do {
            if let error = header.error { throw DataBrowserError.server(error) }
            finish(requestID, result: .success(try decode(data, type: header.type)))
        } catch {
            finish(requestID, result: .failure(error is DataBrowserError ? error : DataBrowserError.invalidResponse))
        }
        return true
    }

    private func decode(_ data: Data, type: String) throws -> DataBrowserResult {
        let decoder = JSONDecoder()
        switch type {
        case "memories_response": return .memories(try decoder.decode(MemoriesResponsePayload.self, from: data).memories)
        case "memory_detail_response": return .detail(MemoryDetail(payload: try decoder.decode(MemoryDetailResponsePayload.self, from: data)))
        case "memory_page_response": return .page(MemoryPage(payload: try decoder.decode(MemoryPageResponsePayload.self, from: data)))
        case "prompt_logs_response":
            let payload = try decoder.decode(PromptLogsResponsePayload.self, from: data)
            return .prompts(payload.runs, hasMore: payload.hasMore)
        default: throw DataBrowserError.invalidResponse
        }
    }

    private func finish(_ id: String, result: Result<DataBrowserResult, Error>) {
        guard let operation = pending.removeValue(forKey: id) else { return }
        operation.timeout.cancel()
        logger.debug("Data browser request completed in \(operation.startedAt.duration(to: .now))", privacy: .public)
        operation.continuation.resume(with: result)
    }
}

private struct CorrelatedRead: Encodable {
    let request: DataBrowserRequest
    let requestID: String

    private enum CodingKeys: String, CodingKey { case requestID = "request_id" }

    func encode(to encoder: Encoder) throws {
        try request.message.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(requestID, forKey: .requestID)
    }
}

private struct ReadHeader: Decodable {
    let type: String
    let requestID: String?
    let error: String?

    var isReadResponse: Bool {
        ["memories_response", "memory_detail_response", "memory_page_response", "prompt_logs_response", "data_read_error"].contains(type)
    }

    private enum CodingKeys: String, CodingKey {
        case type, error
        case requestID = "request_id"
    }
}
