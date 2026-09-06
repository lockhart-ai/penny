import Foundation
import Testing
@testable import PennyServices

@Suite(.serialized)
@MainActor
struct DataBrowserClientTests {
    @Test func currentMemoryContractAndCorrelatedReads() async throws {
        let capture = ReadCapture()
        let client = DataBrowserClient(send: capture.send)
        let task = Task { try await client.read(.memories(query: "coffee")) }
        let request = try await capture.next(0)
        #expect(request["type"] == .string("memories_request"))
        #expect(request["query"] == .string("coffee"))
        let response = try capture.response(to: request, type: "memories_response", fields: ["memories": .array([memoryFixture])])
        #expect(client.receive(response))
        guard case .memories(let memories) = try await task.value else { Issue.record("Wrong result"); return }
        #expect(memories.first?.notificationsEnabled == true)
        #expect(memories.first?.schedule == "FREQ=DAILY")
        #expect(memories.first?.entryCount == 0)
    }

    @Test func reversedResponsesStayWithTheirRequestsAndCancelledRepliesAreDropped() async throws {
        let capture = ReadCapture()
        let client = DataBrowserClient(send: capture.send)
        let first = Task { try await client.read(.memories(query: "first")) }
        let firstRequest = try await capture.next(0)
        let second = Task { try await client.read(.memories(query: "second")) }
        let secondRequest = try await capture.next(1)
        #expect(client.receive(try capture.response(to: secondRequest, type: "memories_response", fields: ["memories": .array([memoryFixture])])))
        guard case .memories(let secondRows) = try await second.value else { Issue.record("Wrong result"); return }
        #expect(secondRows.count == 1)
        first.cancel()
        await #expect(throws: CancellationError.self) { try await first.value }
        #expect(client.receive(try capture.response(to: firstRequest, type: "memories_response", fields: ["memories": .array([])])))
    }

    @Test func missingCorrelationAndServerErrorsFailExplicitly() async throws {
        let capture = ReadCapture()
        let client = DataBrowserClient(send: capture.send)
        let legacy = Task { try await client.read(.memories(query: nil)) }
        _ = try await capture.next(0)
        #expect(client.receive(Data("{\"type\":\"memories_response\",\"memories\":[]}".utf8)))
        await #expect(throws: DataBrowserError.self) { try await legacy.value }
        let failing = Task { try await client.read(.detail(name: "missing", query: nil)) }
        let request = try await capture.next(1)
        #expect(client.receive(try capture.response(to: request, type: "data_read_error", fields: ["error": .string("This memory no longer exists.")])))
        do {
            _ = try await failing.value
            Issue.record("Expected server error")
        } catch {
            #expect(error.localizedDescription == "This memory no longer exists.")
        }
    }

    @Test func timeoutDisconnectMalformedResponseAndSendFailureCompleteRequests() async throws {
        let capture = ReadCapture()
        let timeout = DataBrowserClient(timeout: .milliseconds(10), send: capture.send)
        await #expect(throws: DataBrowserError.self) { try await timeout.read(.memories(query: nil)) }
        let client = DataBrowserClient(send: capture.send)
        let pending = Task { try await client.read(.memories(query: nil)) }
        _ = try await capture.next(1)
        client.disconnect()
        await #expect(throws: DataBrowserError.self) { try await pending.value }
        let malformed = Task { try await client.read(.memories(query: nil)) }
        let request = try await capture.next(2)
        #expect(client.receive(try capture.response(to: request, type: "memories_response", fields: [:])))
        await #expect(throws: DataBrowserError.self) { try await malformed.value }
        let failure = DataBrowserClient { _ in throw DataBrowserError.disconnected }
        await #expect(throws: DataBrowserError.self) { try await failure.read(.memories(query: nil)) }
    }

    @Test func allReadKindsEncodeWithoutMutationMessages() async throws {
        let capture = ReadCapture()
        let client = DataBrowserClient(send: capture.send)
        let inputs: [DataBrowserRequest] = [
            .memories(query: nil), .detail(name: "fixture", query: "match"),
            .page(name: "fixture", section: .entries, offset: 50, query: "match"),
            .prompts(agent: "collector", query: "failure", flagged: true, offset: 25)
        ]
        for (index, input) in inputs.enumerated() {
            let task = Task { try await client.read(input) }
            _ = try await capture.next(index)
            task.cancel()
            await #expect(throws: CancellationError.self) { try await task.value }
        }
        #expect(capture.requests.map { $0["type"] } == [
            .string("memories_request"), .string("memory_detail_request"),
            .string("memory_page_request"), .string("prompt_logs_request")
        ])
        #expect(capture.requests[2]["offset"] == .number(50))
        #expect(capture.requests[3]["flagged_only"] == .bool(true))
    }
}

@MainActor
private final class ReadCapture {
    var requests: [[String: JSONValue]] = []

    func send(_ data: Data) async throws {
        requests.append(try JSONDecoder().decode([String: JSONValue].self, from: data))
    }

    func next(_ index: Int) async throws -> [String: JSONValue] {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while requests.count <= index {
            guard ContinuousClock.now < deadline else { throw DataBrowserError.timedOut }
            await Task.yield()
        }
        return requests[index]
    }

    func response(to request: [String: JSONValue], type: String, fields: [String: JSONValue]) throws -> Data {
        var response = fields
        response["type"] = .string(type)
        response["request_id"] = request["request_id"]
        return try JSONEncoder().encode(response)
    }
}

private let memoryFixture: JSONValue = .object([
    "name": .string("fixture"), "type": .string("collection"), "description": .string("Synthetic notes"),
    "published": .bool(true), "archived": .bool(false), "schedule": .string("FREQ=DAILY"), "entry_count": .number(0)
])
