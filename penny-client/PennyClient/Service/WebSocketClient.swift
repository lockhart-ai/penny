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
    private let logger: any LogService
    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var onReceive: ReceiveHandler?
    private var onFailure: FailureHandler?

    init(
        urlSession: URLSession = URLSession(configuration: .default),
        maximumMessageSize: Int = 20 * 1024 * 1024,
        logger: (any LogService)? = nil
    ) {
        self.urlSession = urlSession
        self.maximumMessageSize = maximumMessageSize
        self.logger = logger ?? OSLogService(category: .webSocket)
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
        logger.debug("sending frame (type=\(Self.frameType(from: data)), \(data.count) bytes)", privacy: .public)
        try await webSocketTask.send(.string(message))
    }

    private func receiveLoop() async {
        while !Task.isCancelled, let webSocketTask {
            do {
                let incomingMessage = try await webSocketTask.receive()
                guard let data = Self.data(from: incomingMessage) else { continue }
                logger.debug("received frame (type=\(Self.frameType(from: data)), \(data.count) bytes)", privacy: .public)
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

    private static func frameType(from data: Data) -> String {
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = object["type"] as? String,
            !type.isEmpty
        else {
            return "unknown"
        }

        return type
    }

}
