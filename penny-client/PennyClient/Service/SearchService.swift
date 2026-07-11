import Foundation
import Metal
import Observation

struct MessageSearchResult: Identifiable {
    let message: ChatMessage
    let distance: Float

    var id: Int { message.id }
    var similarity: Float { max(0, 1 - distance) }
}

enum MessageSearchError: LocalizedError {
    case emptyQuery
    case invalidEmbedding
    case embeddingTimedOut
    case metalUnavailable
    case metalExecutionFailed(String)

    var errorDescription: String? {
        switch self {
        case .emptyQuery:
            return "Enter a search phrase."
        case .invalidEmbedding:
            return "Penny returned an invalid embedding."
        case .embeddingTimedOut:
            return "Penny did not return a search embedding in time."
        case .metalUnavailable:
            return "Semantic search is unavailable on this device."
        case .metalExecutionFailed(let message):
            return "Semantic search failed: \(message)"
        }
    }
}

@MainActor
@Observable
final class SearchService {
    static let maximumResults = 50
    static let minimumSimilarity: Float = 0.24

    private let databaseService: DatabaseService
    private let engine: any CosineDistanceEngine
    private let embeddingTimeout: Duration
    private let requestEmbedding: (String) async throws -> Data
    private var searchGeneration = 0

    var results: [MessageSearchResult] = []
    var isSearching = false
    var hasSearched = false
    var errorMessage: String?

    convenience init(client: PennyService) {
        self.init(client: client, databaseService: .shared)
    }

    convenience init(client: PennyService, databaseService: DatabaseService) {
        self.init(
            databaseService: databaseService,
            engine: MetalCosineDistanceEngine(),
            embeddingTimeout: .seconds(15),
            requestEmbedding: { try await client.requestEmbedding($0) }
        )
    }

    init(
        databaseService: DatabaseService,
        engine: any CosineDistanceEngine,
        embeddingTimeout: Duration = .seconds(15),
        requestEmbedding: @escaping (String) async throws -> Data
    ) {
        self.databaseService = databaseService
        self.engine = engine
        self.embeddingTimeout = embeddingTimeout
        self.requestEmbedding = requestEmbedding
    }

    func search(_ query: String) async {
        searchGeneration += 1
        let generation = searchGeneration
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedQuery.isEmpty else {
            clear(generation: generation)
            return
        }

        isSearching = true
        hasSearched = true
        errorMessage = nil
        defer { finishSearch(generation: generation) }

        let messages = await databaseService.loadMessagesInBackground()
        guard generation == searchGeneration else { return }

        let localMatches = lexicalMatches(in: messages, query: trimmedQuery)
        let candidates = messages.compactMap { model -> (MessageModel, [Float])? in
            guard let embedding = model.embedding,
                  let vector = try? decodeFloat32Vector(embedding) else { return nil }
            return (model, vector)
        }

        guard !candidates.isEmpty else {
            results = localMatches
            return
        }

        do {
            let queryVector = try decodeFloat32Vector(
                try await requestEmbeddingWithTimeout(trimmedQuery)
            )
            guard generation == searchGeneration else { return }

            let compatibleCandidates = candidates.filter { $0.1.count == queryVector.count }
            guard !compatibleCandidates.isEmpty else {
                results = localMatches
                return
            }

            let distances = try engine.distances(
                query: queryVector,
                candidates: compatibleCandidates.map(\.1)
            )
            guard generation == searchGeneration else { return }

            let semanticResults = zip(compatibleCandidates, distances)
                .compactMap { item -> MessageSearchResult? in
                    let candidate = item.0
                    let distance = item.1
                    guard 1 - distance >= Self.minimumSimilarity else { return nil }
                    return MessageSearchResult(
                        message: ChatMessage(model: candidate.0),
                        distance: distance
                    )
                }
                .sorted { lhs, rhs in lhs.distance < rhs.distance }

            results = mergeResults(semanticResults: semanticResults, localMatches: localMatches)
        } catch is CancellationError {
            return
        } catch {
            guard generation == searchGeneration else { return }
            if localMatches.isEmpty {
                results = []
                errorMessage = error.localizedDescription
            } else {
                results = localMatches
            }
        }
    }

    func clear() {
        searchGeneration += 1
        clear(generation: searchGeneration)
    }

    private func clear(generation: Int) {
        guard generation == searchGeneration else { return }
        isSearching = false
        hasSearched = false
        results = []
        errorMessage = nil
    }

    private func finishSearch(generation: Int) {
        guard generation == searchGeneration else { return }
        isSearching = false
    }

    private func requestEmbeddingWithTimeout(_ query: String) async throws -> Data {
        try await withThrowingTaskGroup(of: Data.self) { group in
            group.addTask { try await self.requestEmbedding(query) }
            group.addTask {
                try await Task.sleep(for: self.embeddingTimeout)
                throw MessageSearchError.embeddingTimedOut
            }

            guard let result = try await group.next() else {
                throw MessageSearchError.embeddingTimedOut
            }
            group.cancelAll()
            return result
        }
    }

    private func mergeResults(
        semanticResults: [MessageSearchResult],
        localMatches: [MessageSearchResult]
    ) -> [MessageSearchResult] {
        let semanticIDs = Set(semanticResults.map(\.id))
        return (semanticResults + localMatches.filter { !semanticIDs.contains($0.id) })
            .prefix(Self.maximumResults)
            .map { $0 }
    }

    private func lexicalMatches(in messages: [MessageModel], query: String) -> [MessageSearchResult] {
        let terms = query
            .lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
        guard !terms.isEmpty else { return [] }

        return messages.compactMap { model in
            guard model.embedding == nil else { return nil }
            let content = model.content.lowercased()
            guard terms.allSatisfy({ content.contains($0) }) else { return nil }
            return MessageSearchResult(
                message: ChatMessage(model: model),
                distance: lexicalDistance(content: content, terms: terms)
            )
        }
        .sorted {
            if $0.distance == $1.distance {
                return $0.message.createdAt > $1.message.createdAt
            }
            return $0.distance < $1.distance
        }
        .prefix(Self.maximumResults)
        .map { $0 }
    }

    private func lexicalDistance(content: String, terms: [String]) -> Float {
        guard !content.isEmpty else { return 1 }
        let queryLength = terms.reduce(0) { $0 + $1.count }
        let coverage = min(1, Float(queryLength) / Float(content.count))
        return 1 - max(Self.minimumSimilarity, coverage)
    }

    private func decodeFloat32Vector(_ data: Data) throws -> [Float] {
        guard data.count >= MemoryLayout<Float>.size,
              data.count.isMultiple(of: MemoryLayout<Float>.size) else {
            throw MessageSearchError.invalidEmbedding
        }
        return data.withUnsafeBytes { rawBuffer in
            stride(from: 0, to: data.count, by: MemoryLayout<Float>.size).map {
                rawBuffer.loadUnaligned(fromByteOffset: $0, as: Float.self)
            }
        }
    }
}

protocol CosineDistanceEngine {
    func distances(query: [Float], candidates: [[Float]]) throws -> [Float]
}

private final class MetalCosineDistanceEngine: CosineDistanceEngine {
    private let commandQueue: MTLCommandQueue?
    private let pipeline: MTLComputePipelineState?

    init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "cosineDistance"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            commandQueue = nil
            self.pipeline = nil
            return
        }
        commandQueue = device.makeCommandQueue()
        self.pipeline = pipeline
    }

    func distances(query: [Float], candidates: [[Float]]) throws -> [Float] {
        guard let commandQueue, let pipeline else {
            throw MessageSearchError.metalUnavailable
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MessageSearchError.metalExecutionFailed("Unable to create a command buffer")
        }

        let flattened = candidates.flatMap { $0 }
        let queryBuffer = deviceBuffer(commandQueue: commandQueue, values: query)
        let candidateBuffer = deviceBuffer(commandQueue: commandQueue, values: flattened)
        guard let queryBuffer, let candidateBuffer,
              let outputBuffer = commandQueue.device.makeBuffer(
                length: candidates.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ) else {
            throw MessageSearchError.metalExecutionFailed("Unable to allocate search buffers")
        }

        var dimension = UInt32(query.count)
        var candidateCount = UInt32(candidates.count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidateBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&candidateCount, length: MemoryLayout<UInt32>.size, index: 4)

        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: candidates.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: max(1, width), height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MessageSearchError.metalExecutionFailed(error.localizedDescription)
        }

        return outputBuffer.contents()
            .bindMemory(to: Float.self, capacity: candidates.count)
            .toArray(count: candidates.count)
    }

    private func deviceBuffer(commandQueue: MTLCommandQueue, values: [Float]) -> MTLBuffer? {
        values.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return nil }
            return commandQueue.device.makeBuffer(bytes: baseAddress, length: bytes.count, options: .storageModeShared)
        }
    }
}

private extension UnsafeMutablePointer where Pointee == Float {
    func toArray(count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: self, count: count))
    }
}
