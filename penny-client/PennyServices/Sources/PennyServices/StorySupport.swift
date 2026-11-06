import Foundation

#if DEBUG
/// Debug-only seam for PennyClient's story mode (#2965): story code outside
/// this module needs a `PennyService` that reads fixture messages from an
/// in-memory database and never opens a real websocket, but `WebSocketTransport`,
/// `DatabaseService`'s real initializers, `PennyService`'s dependency-injecting
/// initializer, and the `AgentProgress...` types' memberwise initializers all
/// stay internal — `storyInstance` is the one public entry point that wraps
/// them, rather than widening any of those.
@MainActor
private final class NullWebSocketTransport: WebSocketTransport {
    /// Reports "already connected" so `PennyService.connect()` finds nothing
    /// to do and never attempts a real socket, regardless of `Prefs` state.
    let isConnected = true

    func connect(request: URLRequest, onReceive: @escaping ReceiveHandler, onFailure: @escaping FailureHandler) {}
    func disconnect() {}
    func send(_ data: Data) async throws {}
}

/// One step of a story's in-progress agent run, named by the tool it used —
/// e.g. a web search, a page read, a memory lookup — so a story can describe
/// realistic multi-step activity without constructing `AgentProgressToolItem`
/// directly (its initializer stays internal).
public struct StoryWorkingStep {
    let toolName: String
    let arguments: [String: AgentProgressValue]

    public init(toolName: String, arguments: [String: AgentProgressValue] = [:]) {
        self.toolName = toolName
        self.arguments = arguments
    }
}

public extension PennyService {
    /// A story-mode `PennyService`: `messages` seeded into a private in-memory
    /// database (the real on-disk database is never touched), with
    /// `isConnected`/`isRegistered` and an in-progress run set directly since
    /// no network round trip will ever update them.
    static func storyInstance(
        messages: [ChatMessage],
        isConnected: Bool = false,
        workingSteps: [StoryWorkingStep] = [],
        backgroundRunCount: Int = 0
    ) -> PennyService {
        let databaseService = DatabaseService()
        databaseService.setupForTesting()
        for message in messages {
            databaseService.save(message: MessageModel(message: message))
        }

        let service = PennyService(databaseService: databaseService, prefs: .shared, webSocketClient: NullWebSocketTransport())
        service.isConnected = isConnected
        service.isRegistered = isConnected
        if !workingSteps.isEmpty {
            service.isTyping = true
            let run = makeWorkingRun(from: workingSteps)
            service.agentProgressRuns = [run.id: run]
        }
        for run in makeBackgroundRuns(count: backgroundRunCount) {
            service.agentProgressRuns[run.id] = run
        }
        return service
    }

    private static func makeWorkingRun(from workingSteps: [StoryWorkingStep]) -> AgentProgressRunItem {
        var run = AgentProgressRunItem(id: "story-run", agent: "chat", scope: .foreground)
        run.steps = workingSteps.enumerated().map { index, step in
            var stepItem = AgentProgressStepItem(number: index + 1, maxSteps: 20)
            stepItem.tools = [AgentProgressToolItem(name: step.toolName, arguments: step.arguments)]
            return stepItem
        }
        return run
    }

    /// `count` running background (collector) runs — the Prompts tab badge's
    /// story fixture (#3046). Each is a bare run with no steps: the badge
    /// only counts running background runs, it doesn't render their detail.
    private static func makeBackgroundRuns(count: Int) -> [AgentProgressRunItem] {
        (0..<count).map { index in
            AgentProgressRunItem(id: "story-background-run-\(index)", agent: "collector", scope: .background)
        }
    }
}
#endif
