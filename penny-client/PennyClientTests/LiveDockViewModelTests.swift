import Foundation
import PennyDesign
import SwiftUI
import Testing
import UIKit
@testable import PennyClient
@testable import PennyServices

/// The live dock's lifecycle, its permission row, and the queued-message
/// marking tied to it (#3042, #3043, #3044, #3130) — split out of
/// `MessageViewModelTests` (which was pushing past SwiftLint's type-body
/// length) rather than trimmed, since every one of these view-model
/// behaviors is testable without rendering a view.
@Suite(.serialized)
@MainActor
struct LiveDockViewModelTests {
    /// The live dock's stream (#3042): each step's thinking (if the server
    /// sent one) before its tool rows, oldest step first, and a tool row
    /// reads "running" only while its step is the run's last — never a
    /// per-tool flag the wire protocol has no event for.
    @Test func liveDockItemsOrderThinkingThenToolsAndMarkOnlyTheLastStepRunning() {
        var run = AgentProgressRunItem(id: "progress", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        run.steps[0].thinking = "Checking the forecast."
        run.steps[0].tools.append(AgentProgressToolItem(name: "browse", arguments: [
            "queries": .array([.string("https://www.example.com/article")])
        ]))
        run.steps.append(AgentProgressStepItem(number: 2, maxSteps: nil))
        run.steps[1].tools.append(AgentProgressToolItem(name: "search_emails", arguments: [:]))

        let items = run.liveDockItems
        #expect(items.count == 3)

        guard case .thinking(let thinkingText) = items[0].kind else {
            Issue.record("expected the first item to be the step's thinking")
            return
        }
        #expect(thinkingText == "Checking the forecast.")

        guard case .tool(_, let firstLabel, let firstRunning) = items[1].kind else {
            Issue.record("expected the second item to be step 1's tool")
            return
        }
        #expect(firstLabel == "Read example.com")
        #expect(firstRunning == false)

        guard case .tool(_, let secondLabel, let secondRunning) = items[2].kind else {
            Issue.record("expected the third item to be step 2's tool")
            return
        }
        #expect(secondLabel == "Searched emails")
        #expect(secondRunning)
    }

    /// The generic default (#3042) humanises a tool this switch has no
    /// bespoke case for and appends its first string argument — the
    /// behaviour an unreleased plugin tool gets, never "not covered".
    @Test func liveDockPresentationFormatsSearchesAndTheGenericDefault() {
        let search = AgentProgressToolItem(name: "browse", arguments: ["queries": .array([.string("sample query")])])
        #expect(liveDockPresentation(for: search).label == "Searched \"sample query\"")

        let noArguments = AgentProgressToolItem(name: "custom_tool", arguments: [:])
        #expect(liveDockPresentation(for: noArguments).label == "Custom tool")

        let withArgument = AgentProgressToolItem(name: "check_availability", arguments: ["date": .string("2026-11-10")])
        #expect(liveDockPresentation(for: withArgument).label == "Check availability · 2026-11-10")

        let withURLArgument = AgentProgressToolItem(name: "read_page", arguments: ["url": .string("https://ipma.pt/en/otempo/")])
        #expect(liveDockPresentation(for: withURLArgument).label == "Read page · ipma.pt")
    }

    /// The dock clears on completion (#3042): a foreground run with no
    /// reply shows it, and once the run disappears with nothing to reveal,
    /// the grace window runs out and the dock goes back to hidden.
    @Test func syncDockShowsTheLiveRunAndClearsAfterTheGraceWindowWithNoReply() async {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        var run = AgentProgressRunItem(id: "run-1", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[run.id] = run

        viewModel.syncDock()
        #expect(viewModel.dockPhase == .working)
        #expect(viewModel.dockRun?.id == "run-1")

        viewModel.client.agentProgressRuns.removeValue(forKey: "run-1")
        viewModel.syncDock()
        #expect(viewModel.dockPhase == .working, "stays working through the grace window, not hidden instantly")

        // Poll rather than sleep a fixed time: the grace task may run late on a loaded runner.
        let deadline = ContinuousClock.now + .seconds(10)
        while viewModel.dockPhase != .hidden, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }
        #expect(viewModel.dockPhase == .hidden)
        #expect(viewModel.dockRun == nil)
    }

    /// A second run starting while the dock is still `.resolving` a first
    /// one (that first reply's reveal never got the chance to finish before
    /// new steps arrived) must not leave the new run's content sitting
    /// behind the old resolve (#3042).
    @Test func syncDockRecoversFromAStuckResolvingPhaseWhenANewRunStarts() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        var firstRun = AgentProgressRunItem(id: "run-a", agent: "chat", scope: .foreground)
        firstRun.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[firstRun.id] = firstRun
        viewModel.syncDock()
        let matched = viewModel.beginResolvingIfMatched(appendedMessages: [ChatMessage(
            id: 1, serverID: 1, createdAt: Date(), content: "Done.",
            sourceHint: nil, runID: "run-a", imageAttachments: [], isOutgoing: false
        )])
        #expect(matched)
        #expect(viewModel.dockPhase == .resolving)

        viewModel.client.agentProgressRuns.removeValue(forKey: "run-a")
        var secondRun = AgentProgressRunItem(id: "run-b", agent: "chat", scope: .foreground)
        secondRun.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[secondRun.id] = secondRun

        viewModel.syncDock()

        #expect(viewModel.dockPhase == .working)
        #expect(viewModel.dockRun?.id == "run-b")
    }

    /// A reply carrying the dock's own run id starts the dock's real exit
    /// immediately — never waiting on `run_finished` — and returns `true`
    /// so the view knows there's a reply to reveal once that exit's
    /// `withAnimation(...) completion:` fires; the view drives that
    /// completion in production, so the test calls `dockExitFinished()`
    /// directly rather than waiting on a timer (#3042, #3130).
    @Test func beginResolvingIfMatchedStartsTheDockExitThenDockExitFinishedRevealsTheReply() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        var run = AgentProgressRunItem(id: "run-2", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        let unrelatedReply = ChatMessage(
            id: 98, serverID: 98, createdAt: Date(), content: "Unrelated.",
            sourceHint: nil, runID: "some-other-run", imageAttachments: [], isOutgoing: false
        )
        let reply = ChatMessage(
            id: 99, serverID: 99, createdAt: Date(), content: "Done.",
            sourceHint: nil, runID: "run-2", imageAttachments: [], isOutgoing: false
        )
        let matched = viewModel.beginResolvingIfMatched(appendedMessages: [unrelatedReply, reply])

        #expect(matched)
        #expect(viewModel.dockPhase == .resolving)
        #expect(viewModel.resolvingMessageID == 99)

        viewModel.dockExitFinished()
        #expect(viewModel.dockPhase == .hidden)
        #expect(viewModel.dockRun == nil)
        #expect(viewModel.resolvingMessageID == nil)
    }

    /// Once the dock's exit genuinely finishes and reveals a matched reply,
    /// nothing has reordered it — it was already the conversation's newest
    /// message the whole time it sat excluded from the grid — but nothing
    /// RE-ANCHORS THE SCROLL to it either, so without this it can land
    /// wherever the dock happened to leave the viewport, partially or
    /// fully off screen (#3130). While at the bottom (the common case —
    /// the dock IS the bottom), revealing it gets the same "new content"
    /// handling a live message already gets elsewhere
    /// (`handleDisplayedMessagesChanged`): a scroll-to-bottom request.
    @Test func dockExitFinishedScrollsToTheRevealedReplyWhenAtBottom() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        var run = AgentProgressRunItem(id: "run-6", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        let question = ChatMessage(
            id: 400, serverID: 400, createdAt: Date().addingTimeInterval(-5), content: "Question.",
            sourceHint: nil, imageAttachments: [], isOutgoing: true
        )
        let reply = ChatMessage(
            id: 401, serverID: 401, createdAt: Date(), content: "Done.",
            sourceHint: nil, runID: "run-6", imageAttachments: [], isOutgoing: false
        )
        viewModel.displayedMessages = [question, reply]
        #expect(viewModel.isAtBottom)

        let matched = viewModel.beginResolvingIfMatched(appendedMessages: [reply])
        #expect(matched)
        let scrollRequestBeforeReveal = viewModel.scrollToBottomRequest

        viewModel.dockExitFinished()

        #expect(viewModel.displayedMessages.last?.id == 401, "the reply is already the newest message — nothing reorders it")
        #expect(viewModel.scrollToBottomRequest == scrollRequestBeforeReveal + 1, "revealing it scrolls to it, same as a live message arriving")
    }

    /// Scrolled up reading older messages when the reveal happens, the
    /// reader's position must not be yanked out from under them — it gets
    /// the same "show new messages" pill handling a live message arriving
    /// while scrolled up already gets (#3130).
    @Test func dockExitFinishedMarksHiddenNewMessagesWhenRevealingAReplyWhileScrolledUp() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        var run = AgentProgressRunItem(id: "run-7", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        let reply = ChatMessage(
            id: 402, serverID: 402, createdAt: Date(), content: "Done.",
            sourceHint: nil, runID: "run-7", imageAttachments: [], isOutgoing: false
        )
        viewModel.displayedMessages = [reply]
        viewModel.updateBottomVisibility(false)

        let matched = viewModel.beginResolvingIfMatched(appendedMessages: [reply])
        #expect(matched)
        let scrollRequestBeforeReveal = viewModel.scrollToBottomRequest

        viewModel.dockExitFinished()

        #expect(viewModel.scrollToBottomRequest == scrollRequestBeforeReveal, "scrolled up — never yank the reader down")
        #expect(viewModel.hasHiddenNewMessages)
    }

    /// `run_finished` landing while a matched reply's exit is genuinely in
    /// flight must NOT cut that exit short (#3130): the reply and the
    /// run's end arrive close together, often reply first — `syncDock()`
    /// seeing no live run while `dockPhase == .resolving` must leave it
    /// alone, or the dock's real exit animation gets cut short and the
    /// reply pops in early, defeating the sequencing this issue exists to
    /// fix. Only the exit animation's own completion, or
    /// `finishLeftoverResolveOnAppear()` below, may finish a resolve.
    @Test func syncDockDoesNotFinishAResolveStillInFlightWhenTheRunEnds() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        var run = AgentProgressRunItem(id: "run-3", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        let reply = ChatMessage(
            id: 199, serverID: 199, createdAt: Date(), content: "Done.",
            sourceHint: nil, runID: "run-3", imageAttachments: [], isOutgoing: false
        )
        let matched = viewModel.beginResolvingIfMatched(appendedMessages: [reply])
        #expect(matched)
        #expect(viewModel.dockPhase == .resolving)

        // `run_finished` lands a beat after the reply, while the exit
        // animation is still genuinely running.
        viewModel.client.agentProgressRuns.removeValue(forKey: "run-3")
        viewModel.syncDock()

        #expect(viewModel.dockPhase == .resolving, "the in-flight exit must not be cut short")
        #expect(viewModel.dockRun?.id == "run-3")
        #expect(viewModel.resolvingMessageID == 199)
    }

    /// The safety net for a resolve truly left stuck (#3130): no new run
    /// ever starts, and the view that would have run the exit animation's
    /// `withAnimation(...) completion:` was never on screen to fire it
    /// (backgrounded, a different tab selected when the reply landed).
    /// `MessageView`'s `.onAppear` calls this alongside `syncDock()` every
    /// time the chat tab (re)appears — e.g. switching back from Memory —
    /// finishing the stuck exit directly instead of leaving the reply
    /// excluded from the grid indefinitely.
    @Test func finishLeftoverResolveOnAppearFinishesAStuckResolve() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        var run = AgentProgressRunItem(id: "run-4", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        let reply = ChatMessage(
            id: 299, serverID: 299, createdAt: Date(), content: "Done.",
            sourceHint: nil, runID: "run-4", imageAttachments: [], isOutgoing: false
        )
        let matched = viewModel.beginResolvingIfMatched(appendedMessages: [reply])
        #expect(matched)
        #expect(viewModel.dockPhase == .resolving)

        // The run already left `agentProgressRuns`, and the view reappears
        // (switching back to the chat tab) before any completion ever fired.
        viewModel.client.agentProgressRuns.removeValue(forKey: "run-4")
        viewModel.syncDock()
        #expect(viewModel.dockPhase == .resolving, "still stuck — syncDock alone never finishes an in-flight resolve")

        viewModel.finishLeftoverResolveOnAppear()

        #expect(viewModel.dockPhase == .hidden)
        #expect(viewModel.dockRun == nil)
        #expect(viewModel.resolvingMessageID == nil)
    }

    /// A no-op outside `.resolving` (#3130) — called unconditionally on
    /// every `.onAppear`, so it must never disturb a live `.working` dock
    /// or an already-`.hidden` one.
    @Test func finishLeftoverResolveOnAppearIsANoOpWhileWorking() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        var run = AgentProgressRunItem(id: "run-5", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[run.id] = run
        viewModel.syncDock()
        #expect(viewModel.dockPhase == .working)

        viewModel.finishLeftoverResolveOnAppear()

        #expect(viewModel.dockPhase == .working)
        #expect(viewModel.dockRun?.id == "run-5")
    }

    /// The dock's permission row (#3044) shows the server's pending ask
    /// only while a foreground run is live — never while the dock is
    /// hidden, which is the Settings-only path (behaviour bullet 3).
    @Test func dockPermissionShowsThePendingPromptOnlyWhileTheDockIsWorking() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        viewModel.client.permissionPrompt = PermissionPrompt(requestID: "req-1", domain: "example.com", url: "https://example.com")

        #expect(viewModel.dockPermission == nil, "no live run — Settings is the only surface")

        var run = AgentProgressRunItem(id: "run-1", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        #expect(viewModel.dockPermission?.domain == "example.com")
    }

    /// Answering from the dock sends the existing decision and turns the
    /// pending row into a normal resolved stream item — the SAME `.tool`
    /// kind every other dock row uses (behaviour bullet 2).
    @Test func decidePermissionSendsTheDecisionAndResolvesTheRowIntoATrailingStreamItem() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        viewModel.client.permissionPrompt = PermissionPrompt(requestID: "req-1", domain: "example.com", url: "https://example.com")
        var run = AgentProgressRunItem(id: "run-1", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        viewModel.decidePermission(allowed: true)

        #expect(viewModel.client.permissionPrompt == nil, "the decision clears the pending prompt, same as Settings")
        #expect(viewModel.dockPermission == nil, "no pending row left to ask about")

        guard case .tool(let symbol, let label, let running) = viewModel.dockStreamItems.last?.kind else {
            Issue.record("expected the resolved permission to trail the stream as a tool row")
            return
        }
        #expect(symbol == "lock.fill")
        #expect(label == "Allowed example.com")
        #expect(running == false)

        viewModel.client.agentProgressRuns.removeValue(forKey: "run-1")
        viewModel.dockExitFinished()
        #expect(viewModel.dockStreamItems.isEmpty, "the resolved row lives only as long as its own run's dock")
    }

    /// A message sent while `dockPhase == .working` is sent immediately
    /// (#2985: the server's turn inbox is what holds and injects it, not
    /// the client) but shown in the queued style until picked up — and
    /// the dock's queued count tracks every message still marked (#3043).
    @Test func sendDraftWhileWorkingMarksEachSentMessageAsQueued() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        var run = AgentProgressRunItem(id: "run-queue", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        viewModel.draftMessage = "Oh, and grab a bag of the dark roast too?"
        viewModel.sendDraft()
        viewModel.draftMessage = "Actually the decaf, if they have it."
        viewModel.sendDraft()

        #expect(client.messages.map(\.content) == [
            "Oh, and grab a bag of the dark roast too?",
            "Actually the decaf, if they have it."
        ])
        let sentIDs = client.messages.map(\.id)
        #expect(sentIDs.allSatisfy(viewModel.isMessageQueued))
        #expect(viewModel.queuedMessageIDs.count == 2)
    }

    /// The queued mark clears the moment the run's next `step_started`
    /// arrives — that's the server having folded the message into the
    /// running turn (#2985) — never waiting for the whole run to finish.
    @Test func queuedMarkClearsOnTheRunsNextStepStarted() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        var run = AgentProgressRunItem(id: "run-queue", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        viewModel.draftMessage = "Oh, and grab a bag of the dark roast too?"
        viewModel.sendDraft()
        let sentID = client.messages[0].id
        #expect(viewModel.isMessageQueued(sentID))

        run.steps.append(AgentProgressStepItem(number: 2, maxSteps: 20))
        client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        #expect(viewModel.isMessageQueued(sentID) == false)
        #expect(viewModel.queuedMessageIDs.isEmpty)
    }

    /// The queued mark also clears when the run ends with no further
    /// step — the message already reached Penny's turn one way or
    /// another (#2985) by the time it's gone.
    @Test func queuedMarkClearsWhenTheRunEnds() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        var run = AgentProgressRunItem(id: "run-queue", agent: "chat", scope: .foreground)
        run.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        client.agentProgressRuns[run.id] = run
        viewModel.syncDock()

        viewModel.draftMessage = "Oh, and grab a bag of the dark roast too?"
        viewModel.sendDraft()
        let sentID = client.messages[0].id
        #expect(viewModel.isMessageQueued(sentID))

        client.agentProgressRuns.removeValue(forKey: "run-queue")
        viewModel.syncDock()

        #expect(viewModel.isMessageQueued(sentID) == false)
    }

    /// A message sent while Penny is idle (no foreground run) never shows
    /// queued — only a busy dock makes sending ambiguous enough to need
    /// the mark.
    @Test func sendDraftWhileIdleNeverMarksAsQueued() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        #expect(viewModel.dockPhase == .hidden)

        viewModel.draftMessage = "What roast works best for a moka pot?"
        viewModel.sendDraft()

        #expect(client.messages.count == 1)
        #expect(viewModel.queuedMessageIDs.isEmpty)
        #expect(viewModel.isMessageQueued(client.messages[0].id) == false)
    }
}
