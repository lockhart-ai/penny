import Foundation

@MainActor
protocol WebSocketTransport {
    typealias ReceiveHandler = (Data) -> Void
    typealias FailureHandler = (Error) -> Void

    var isConnected: Bool { get }

    func connect(
        request: URLRequest,
        onReceive: @escaping ReceiveHandler,
        onFailure: @escaping FailureHandler
    )
    func disconnect()
    func send(_ data: Data) async throws
}

@MainActor
final class WebSocketClient: WebSocketTransport {
    private let urlSession: URLSession
    private let maximumMessageSize: Int
    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var onReceive: ReceiveHandler?
    private var onFailure: FailureHandler?

    init(
        urlSession: URLSession = URLSession(configuration: .default),
        maximumMessageSize: Int = 20 * 1024 * 1024
    ) {
        self.urlSession = urlSession
        self.maximumMessageSize = maximumMessageSize
    }

    var isConnected: Bool {
        webSocketTask != nil
    }

    func connect(
        request: URLRequest,
        onReceive: @escaping ReceiveHandler,
        onFailure: @escaping FailureHandler
    ) {
        guard webSocketTask == nil else { return }

        self.onReceive = onReceive
        self.onFailure = onFailure

        let task = urlSession.webSocketTask(with: request)
        task.maximumMessageSize = maximumMessageSize
        webSocketTask = task
        task.resume()

        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }
    }

    func disconnect() {
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        onReceive = nil
        onFailure = nil
    }

    func send(_ data: Data) async throws {
        guard let webSocketTask else { return }
        guard let message = String(data: data, encoding: .utf8) else { return }
        try await webSocketTask.send(.string(message))
    }

    private func receiveLoop() async {
        while !Task.isCancelled, let webSocketTask {
            do {
                let incomingMessage = try await webSocketTask.receive()
                guard let data = Self.data(from: incomingMessage) else { continue }
                debugLogFrame("received frame (\(data.count) bytes)")
                onReceive?(data)
            } catch {
                guard !Task.isCancelled else { return }
                onFailure?(error)
                return
            }
        }
    }

    private static func data(from message: URLSessionWebSocketTask.Message) -> Data? {
        switch message {
        case .data(let messageData):
            return messageData
        case .string(let messageString):
            return Data(messageString.utf8)
        @unknown default:
            return nil
        }
    }

    /// Logs a short, non-sensitive frame summary in debug builds only. Never logs frame
    /// contents: outbound frames can carry device secrets, APNs tokens, or chat content.
    private func debugLogFrame(_ summary: @autoclosure () -> String) {
        #if DEBUG
        print("[WebSocketClient] \(summary())")
        #endif
    }
}
