import Foundation
import Testing
@testable import PennyClient

@Suite(.serialized)
@MainActor
struct SearchServiceTests {
    @Test func emptyQueryClearsResultsWithoutRequestingTheServer() async {
        let recorder = EmbeddingRequestRecorder()
        let service = SearchService(
            databaseService: configuredDatabase(),
            engine: SearchTestDistanceEngine(),
            requestEmbedding: recorder.request
        )
        service.errorMessage = "old error"

        await service.search("   ")

        #expect(service.results.isEmpty)
        #expect(service.errorMessage == nil)
        #expect(service.isSearching == false)
        #expect(service.hasSearched == false)
        #expect(recorder.requests.isEmpty)
    }

    @Test func searchIgnoresUnembeddedMessagesWithoutRequestingTheServer() async {
        let database = configuredDatabase()
        database.save(message: MessageModel(
            id: 1,
            serverID: 1,
            createdAt: Date(timeIntervalSince1970: 1),
            content: "Coffee beans are in the pantry",
            sourceHint: "Chat",
            imageAttachmentDataURLs: [],
            isOutgoing: false
        ))
        let recorder = EmbeddingRequestRecorder(error: PennyEmbeddingError.disconnected)
        let service = SearchService(
            databaseService: database,
            engine: SearchTestDistanceEngine(),
            requestEmbedding: recorder.request
        )

        await service.search("coffee pantry")

        #expect(service.results.isEmpty)
        #expect(service.errorMessage == nil)
        #expect(service.isSearching == false)
        #expect(service.hasSearched)
        #expect(recorder.requests.isEmpty)
    }

    @Test func searchAppliesSelectedMessageFilter() async {
        let database = configuredDatabase()
        database.save(message: MessageModel(
            id: 1,
            serverID: 1,
            createdAt: Date(timeIntervalSince1970: 1),
            content: "Coffee chat note",
            sourceHint: "Chat",
            imageAttachmentDataURLs: [],
            isOutgoing: false,
            embedding: floatData([1, 0])
        ))
        database.save(message: MessageModel(
            id: 2,
            serverID: 2,
            createdAt: Date(timeIntervalSince1970: 2),
            content: "Coffee schedule note",
            sourceHint: "Schedule",
            imageAttachmentDataURLs: [],
            isOutgoing: false,
            embedding: floatData([1, 0])
        ))
        let recorder = EmbeddingRequestRecorder()
        let service = SearchService(
            databaseService: database,
            engine: SearchTestDistanceEngine(),
            requestEmbedding: recorder.request
        )

        let search = Task { await service.search("coffee", filter: .schedule) }
        await waitFor { recorder.requests == ["coffee"] }
        recorder.resumeRequest(at: 0, with: floatData([1, 0]))
        await search.value

        #expect(service.results.map(\.message.content) == ["Coffee schedule note"])
        #expect(service.hasSearched)
    }

    @Test func submittedSearchWithNoMatchesRecordsThatSearchOccurred() async {
        let database = configuredDatabase()
        database.save(message: MessageModel(
            id: 1,
            serverID: 1,
            createdAt: Date(timeIntervalSince1970: 1),
            content: "Coffee beans are in the pantry",
            sourceHint: "Chat",
            imageAttachmentDataURLs: [],
            isOutgoing: false
        ))
        let service = SearchService(
            databaseService: database,
            engine: SearchTestDistanceEngine(),
            requestEmbedding: EmbeddingRequestRecorder().request
        )

        await service.search("missing")

        #expect(service.results.isEmpty)
        #expect(service.errorMessage == nil)
        #expect(service.hasSearched)
    }

    @Test func embeddingRequestTimesOutInsteadOfSearchingForever() async {
        let database = configuredDatabase()
        database.save(message: MessageModel(
            id: 1,
            serverID: 1,
            createdAt: Date(timeIntervalSince1970: 1),
            content: "Vector only result",
            sourceHint: "Chat",
            imageAttachmentDataURLs: [],
            isOutgoing: false,
            embedding: floatData([1, 0])
        ))
        let recorder = EmbeddingRequestRecorder(shouldHang: true)
        let service = SearchService(
            databaseService: database,
            engine: SearchTestDistanceEngine(),
            embeddingTimeout: .milliseconds(20),
            requestEmbedding: recorder.request
        )

        await service.search("vector")

        #expect(service.results.isEmpty)
        #expect(service.errorMessage == MessageSearchError.embeddingTimedOut.localizedDescription)
        #expect(service.isSearching == false)
        #expect(service.hasSearched)
        #expect(recorder.requests == ["vector"])
    }

    @Test func staleSearchDoesNotClearLoadingStateForNewerSearch() async {
        let database = configuredDatabase()
        database.save(message: MessageModel(
            id: 1,
            serverID: 1,
            createdAt: Date(timeIntervalSince1970: 1),
            content: "First vector result",
            sourceHint: "Chat",
            imageAttachmentDataURLs: [],
            isOutgoing: false,
            embedding: floatData([1, 0])
        ))
        let recorder = EmbeddingRequestRecorder()
        let service = SearchService(
            databaseService: database,
            engine: SearchTestDistanceEngine(distances: [0.05]),
            requestEmbedding: recorder.request
        )

        let firstSearch = Task { await service.search("first") }
        await waitFor { recorder.requests.count == 1 }
        let secondSearch = Task { await service.search("second") }
        await waitFor { recorder.requests.count == 2 }

        recorder.resumeRequest(at: 0, with: floatData([1, 0]))
        await waitFor { service.isSearching }
        #expect(service.isSearching)

        recorder.resumeRequest(at: 1, with: floatData([1, 0]))
        await firstSearch.value
        await secondSearch.value

        #expect(service.isSearching == false)
        #expect(service.results.map(\.message.content) == ["First vector result"])
    }

    @Test func clearCancelsVisibleStateImmediately() async {
        let database = configuredDatabase()
        database.save(message: MessageModel(
            id: 1,
            serverID: 1,
            createdAt: Date(timeIntervalSince1970: 1),
            content: "Vector only result",
            sourceHint: "Chat",
            imageAttachmentDataURLs: [],
            isOutgoing: false,
            embedding: floatData([1, 0])
        ))
        let recorder = EmbeddingRequestRecorder(shouldHang: true)
        let service = SearchService(
            databaseService: database,
            engine: SearchTestDistanceEngine(),
            embeddingTimeout: .seconds(1),
            requestEmbedding: recorder.request
        )

        let search = Task { await service.search("vector") }
        await waitFor { recorder.requests.count == 1 }
        service.clear()
        search.cancel()
        await search.value

        #expect(service.results.isEmpty)
        #expect(service.errorMessage == nil)
        #expect(service.isSearching == false)
        #expect(service.hasSearched == false)
    }
}

private struct SearchTestDistanceEngine: CosineDistanceEngine {
    var distances: [Float] = []

    func distances(query: [Float], candidates: [[Float]]) throws -> [Float] {
        if distances.isEmpty {
            return Array(repeating: 0, count: candidates.count)
        }
        return distances
    }
}

@MainActor
private final class EmbeddingRequestRecorder {
    private var continuations: [CheckedContinuation<Data, Error>] = []
    private let error: Error?
    private let shouldHang: Bool
    var requests: [String] = []

    init(error: Error? = nil, shouldHang: Bool = false) {
        self.error = error
        self.shouldHang = shouldHang
    }

    func request(_ query: String) async throws -> Data {
        requests.append(query)
        if let error {
            throw error
        }
        if shouldHang {
            try await Task.sleep(for: .seconds(60))
            throw CancellationError()
        }
        return try await withCheckedThrowingContinuation { continuation in
            continuations.append(continuation)
        }
    }

    func resumeRequest(at index: Int, with data: Data) {
        guard continuations.indices.contains(index) else {
            Issue.record("No embedding request continuation at index \(index)")
            return
        }
        continuations[index].resume(returning: data)
    }
}

private func floatData(_ values: [Float]) -> Data {
    var values = values
    return Data(bytes: &values, count: values.count * MemoryLayout<Float>.size)
}

@MainActor
private func waitFor(
    _ condition: @escaping @MainActor () -> Bool,
    sourceLocation: SourceLocation = #_sourceLocation
) async {
    for _ in 0..<100 {
        if condition() { return }
        try? await Task.sleep(for: .milliseconds(10))
    }
    Issue.record("Timed out waiting for condition", sourceLocation: sourceLocation)
}
