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
    let toolName: String?
    let arguments: [String: AgentProgressValue]
    /// The step's reasoning (#3040), shown in the live dock (#3042) as the
    /// italic line above the step's tool row(s). `nil` renders no thinking
    /// line for that step, matching a server frame with no `thinking` field.
    let thinking: String?

    /// `toolName` is optional so a story can describe a step whose
    /// thinking arrived with no tool call yet — the live dock's very first
    /// frame (#3042, `chat-working-start`): a step with thinking and no
    /// tool renders only the thinking line, the same as a real in-flight
    /// step the server hasn't sent `tools_started` for yet.
    public init(toolName: String? = nil, arguments: [String: AgentProgressValue] = [:], thinking: String? = nil) {
        self.toolName = toolName
        self.arguments = arguments
        self.thinking = thinking
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
        workingElapsedSeconds: Int = 0,
        backgroundRunCount: Int = 0,
        pendingPermissionDomain: String? = nil
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
            let run = makeWorkingRun(from: workingSteps, elapsedSeconds: workingElapsedSeconds)
            service.agentProgressRuns = [run.id: run]
        }
        for run in makeBackgroundRuns(count: backgroundRunCount) {
            service.agentProgressRuns[run.id] = run
        }
        if let pendingPermissionDomain {
            service.permissionPrompt = PermissionPrompt(
                requestID: "story-permission",
                domain: pendingPermissionDomain,
                url: "https://\(pendingPermissionDomain)"
            )
        }
        return service
    }

    /// `elapsedSeconds` backdates `startedAt` so the live dock's header
    /// reads a fixed "t s" at render time (#3042) — a story is screenshotted
    /// moments after this runs, so a `startedAt` of "now" would read ~0 s.
    private static func makeWorkingRun(from workingSteps: [StoryWorkingStep], elapsedSeconds: Int) -> AgentProgressRunItem {
        var run = AgentProgressRunItem(id: "story-run", agent: "chat", scope: .foreground)
        run.startedAt = Date().addingTimeInterval(-Double(elapsedSeconds))
        run.steps = workingSteps.enumerated().map { index, step in
            var stepItem = AgentProgressStepItem(number: index + 1, maxSteps: 20)
            if let toolName = step.toolName {
                stepItem.tools = [AgentProgressToolItem(name: toolName, arguments: step.arguments)]
            }
            stepItem.thinking = step.thinking
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
