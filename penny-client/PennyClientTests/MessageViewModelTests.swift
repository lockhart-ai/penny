import Foundation
import PennyDesign
import SwiftUI
import Testing
import UIKit
@testable import PennyClient
@testable import PennyServices

@Suite(.serialized)
@MainActor
struct MessageViewModelTests {
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
    /// reply shows it, and once the run disappears with nothing to morph
    /// into, the grace window runs out and the dock goes back to hidden.
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
    /// one (that first reply's morph never got its full duration to finish
    /// before new steps arrived) must not leave the new run's content
    /// sitting behind the old resolve's faded opacity (#3042).
    @Test func syncDockRecoversFromAStuckResolvingPhaseWhenANewRunStarts() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        var firstRun = AgentProgressRunItem(id: "run-a", agent: "chat", scope: .foreground)
        firstRun.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[firstRun.id] = firstRun
        viewModel.syncDock()
        viewModel.beginResolvingIfMatched(appendedMessages: [ChatMessage(
            id: 1, serverID: 1, createdAt: Date(), content: "Done.",
            sourceHint: nil, runID: "run-a", imageAttachments: [], isOutgoing: false
        )])
        #expect(viewModel.dockPhase == .resolving)

        viewModel.client.agentProgressRuns.removeValue(forKey: "run-a")
        var secondRun = AgentProgressRunItem(id: "run-b", agent: "chat", scope: .foreground)
        secondRun.steps.append(AgentProgressStepItem(number: 1, maxSteps: 20))
        viewModel.client.agentProgressRuns[secondRun.id] = secondRun

        viewModel.syncDock()

        #expect(viewModel.dockPhase == .working)
        #expect(viewModel.dockRun?.id == "run-b")
    }

    /// A reply carrying the dock's own run id starts the resolve morph
    /// immediately — never waiting on `run_finished` — and the dock clears
    /// once the morph's duration elapses (#3042).
    @Test func beginResolvingIfMatchedStartsTheMorphThenClearsTheDock() async {
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
        viewModel.beginResolvingIfMatched(appendedMessages: [unrelatedReply, reply])

        #expect(viewModel.dockPhase == .resolving)
        #expect(viewModel.resolvingMessageID == 99)

        try? await Task.sleep(for: .milliseconds(500))
        #expect(viewModel.dockPhase == .hidden)
        #expect(viewModel.dockRun == nil)
        #expect(viewModel.resolvingMessageID == nil)
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
        viewModel.finishDock()
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

    @Test func messageLayoutDefaultsToCurrentMessageStyle() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        #expect(viewModel.selectedMessageLayout == .message)
    }

    @Test func messageLayoutCanSwitchBetweenAvailableLayouts() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        viewModel.selectedMessageLayout = .compact
        #expect(viewModel.selectedMessageLayout == .compact)

        viewModel.selectedMessageLayout = .media
        #expect(viewModel.selectedMessageLayout == .media)
    }

    @Test func composerFocusReturnsToMessageLayoutWithoutRequestingScroll() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        viewModel.selectedMessageLayout = .media
        let previousScrollRequest = viewModel.scrollToBottomRequest

        let didChangeLayout = viewModel.prepareComposerFocus()

        #expect(didChangeLayout)
        #expect(viewModel.selectedMessageLayout == .message)
        #expect(viewModel.scrollToBottomRequest == previousScrollRequest)
    }

    @Test func composerFocusDoesNotRequestScrollWhenAlreadyUsingMessageLayout() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        let previousScrollRequest = viewModel.scrollToBottomRequest

        let didChangeLayout = viewModel.prepareComposerFocus()

        #expect(didChangeLayout == false)
        #expect(viewModel.selectedMessageLayout == .message)
        #expect(viewModel.scrollToBottomRequest == previousScrollRequest)
    }

    @Test func composerPresentationTracksEveryConversationState() {
        let viewModel = MessageView.ViewModel(
            client: PennyWebSocketClient(
                databaseService: configuredDatabase(),
                prefs: configuredPrefs()
            )
        )

        viewModel.client.conversationState = .elicit
        #expect(viewModel.composerPresentation.placeholder == "Teach Penny")
        #expect(viewModel.composerPresentation.tint == Color.purple.opacity(0.18))

        viewModel.client.conversationState = .learn
        #expect(viewModel.composerPresentation.placeholder == "Approve or correct")
        #expect(viewModel.composerPresentation.tint == Color.blue.opacity(0.16))

        viewModel.client.conversationState = .request
        #expect(viewModel.composerPresentation.placeholder == "Add the missing detail")
        #expect(viewModel.composerPresentation.tint == Color.orange.opacity(0.18))

        for state in [ConversationStateLabel.idle, .apply] {
            viewModel.client.conversationState = state
            #expect(viewModel.composerPresentation.placeholder == "Message")
            #expect(viewModel.composerPresentation.tint == nil)
        }

        viewModel.client.conversationState = nil
        #expect(viewModel.composerPresentation.placeholder == "Message")
        #expect(viewModel.composerPresentation.tint == nil)
    }

    @Test func startupLoadsLatestPageOnly() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...5)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)

        await viewModel.connect()

        #expect(viewModel.displayedMessages.map(\.id) == [4, 5])
        #expect(viewModel.hasMoreOlderMessages)
    }

    @Test func scrollingUpPrependsOlderMessages() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...5)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()

        let anchorID = viewModel.reserveOlderMessageLoad()
        let didLoad = await viewModel.loadReservedOlderMessages()

        #expect(anchorID == 4)
        #expect(didLoad)
        #expect(viewModel.displayedMessages.map(\.id) == [2, 3, 4, 5])
        #expect(viewModel.hasMoreOlderMessages)
    }

    @Test func olderMessageLoadReservationIgnoresDuplicateTriggers() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...5)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()

        let firstAnchorID = viewModel.reserveOlderMessageLoad()
        let secondAnchorID = viewModel.reserveOlderMessageLoad()
        let didLoad = await viewModel.loadReservedOlderMessages()
        let thirdAnchorID = viewModel.reserveOlderMessageLoad()
        viewModel.finishOlderMessageScrollRestoration()
        let fourthAnchorID = viewModel.reserveOlderMessageLoad()

        #expect(firstAnchorID == 4)
        #expect(secondAnchorID == nil)
        #expect(didLoad)
        #expect(thirdAnchorID == nil)
        #expect(fourthAnchorID == 2)
        #expect(viewModel.displayedMessages.map(\.id) == [2, 3, 4, 5])
    }

    @Test func scrollRequestCanSkipLayoutSettling() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        viewModel.requestScrollToBottom(shouldSettleLayout: false)

        #expect(viewModel.shouldSettleScrollToBottom == false)
        #expect(viewModel.scrollToBottomRequest == 1)
    }

    @Test func bottomScrollRequestBlocksOlderPagingUntilScrollLands() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...5)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()

        viewModel.requestScrollToBottom()
        let anchorID = viewModel.reserveOlderMessageLoad()

        #expect(anchorID == nil)
    }

    @Test func filterChangeCancelsReservedOlderMessageLoad() async {
        let database = configuredDatabase()
        saveFilterFixture(in: database)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()

        let anchorID = viewModel.reserveOlderMessageLoad()
        viewModel.selectedMessageFilter = .penny
        await viewModel.waitForPaging()
        let didLoad = await viewModel.loadReservedOlderMessages()

        #expect(anchorID == 7)
        #expect(didLoad == false)
        #expect(viewModel.displayedMessages.map(\.id) == [2, 5])
        #expect(viewModel.hasMoreOlderMessages)
    }

    @Test func filterReloadRequestsBottomScrollAndDisablesOlderPagingUntilScrollLands() async {
        let database = configuredDatabase()
        saveFilterFixture(in: database)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 2)
        await viewModel.connect()
        viewModel.enableOlderPaging()
        let previousScrollRequest = viewModel.scrollToBottomRequest

        viewModel.selectedMessageFilter = .penny
        await viewModel.waitForPaging()
        let blockedAnchorID = viewModel.reserveOlderMessageLoad()
        viewModel.enableOlderPaging()
        let unblockedAnchorID = viewModel.reserveOlderMessageLoad()

        #expect(viewModel.scrollToBottomRequest == previousScrollRequest + 1)
        #expect(viewModel.hasMoreOlderMessages)
        #expect(blockedAnchorID == nil)
        #expect(unblockedAnchorID == 2)
    }

    @Test func changingFiltersReloadsLatestFilteredPage() async {
        let database = configuredDatabase()
        saveFilterFixture(in: database)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()

        viewModel.selectedMessageFilter = .penny
        await viewModel.waitForPaging()
        #expect(viewModel.displayedMessages.map(\.id) == [1, 2, 5])

        viewModel.selectedMessageFilter = .chat
        await viewModel.waitForPaging()
        #expect(viewModel.displayedMessages.map(\.id) == [4, 8])
        #expect(viewModel.hasMoreOlderMessages == false)
    }

    @Test func sendDraftTrimsMessageClearsDraftAndFilter() async {
        let database = configuredDatabase()
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        viewModel.selectedMessageFilter = .penny
        await viewModel.waitForPaging()
        viewModel.updateBottomVisibility(false)
        let previousScrollRequest = viewModel.scrollToBottomRequest
        viewModel.draftMessage = "  hello Penny  "

        viewModel.sendDraft()
        await viewModel.waitForPaging()

        #expect(viewModel.selectedMessageFilter == .all)
        #expect(viewModel.draftMessage.isEmpty)
        #expect(viewModel.displayedMessages.map(\.content) == ["hello Penny"])
        #expect(database.loadMessages().first?.content == "hello Penny")
        #expect(viewModel.scrollToBottomRequest > previousScrollRequest)
        #expect(viewModel.isAtBottom)
    }

    @Test func sendDraftAppendsReplyContextAndClearsReplyState() async {
        let database = configuredDatabase()
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        let originalMessage = ChatMessage(model: testMessage(
            id: 42,
            content: "First line\n\nSecond line",
            sourceHint: "Chat"
        ))
        viewModel.startReply(to: originalMessage)
        viewModel.draftMessage = "  That makes sense  "

        viewModel.sendDraft()

        #expect(viewModel.replyMessage == nil)
        #expect(viewModel.draftMessage.isEmpty)
        #expect(database.loadMessages().last?.content == """
        That makes sense

        In reply to Chat:
        > First line Second line
        """)
    }

    @Test func cancelReplyClearsReplyState() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))
        let originalMessage = ChatMessage(model: testMessage(id: 42, content: "Original", sourceHint: "Chat"))

        viewModel.startReply(to: originalMessage)
        viewModel.cancelReply()

        #expect(viewModel.replyMessage == nil)
    }

    @Test func sendDraftIgnoresWhitespaceOnlyDraft() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.selectedMessageFilter = .penny
        viewModel.draftMessage = "   \n  "

        viewModel.sendDraft()

        #expect(viewModel.selectedMessageFilter == .penny)
        #expect(viewModel.draftMessage == "   \n  ")
        #expect(client.messages.isEmpty)
    }

    @Test func sendDraftSendsPendingPhotoWithTheMessageAndClearsIt() async throws {
        let database = configuredDatabase()
        let client = PennyWebSocketClient(databaseService: database, prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.draftMessage = "  What's this bird?  "
        viewModel.pendingImage = solidTestImage()

        viewModel.sendDraft()

        #expect(viewModel.draftMessage.isEmpty)
        #expect(viewModel.pendingImage == nil)
        let sent = try #require(client.messages.last)
        #expect(sent.content == "What's this bird?")
        #expect(sent.imageAttachments.count == 1)
        #expect(sent.imageAttachmentDataURLs.count == 1)
        #expect(sent.imageAttachmentDataURLs.first?.hasPrefix("data:image/jpeg;base64,") == true)
        #expect(database.loadMessages().last?.imageAttachmentDataURLs.count == 1)
    }

    @Test func sendDraftSendsImageOnlyMessage() async throws {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.draftMessage = "   "
        viewModel.pendingImage = solidTestImage()

        viewModel.sendDraft()

        let sent = try #require(client.messages.last)
        #expect(sent.content.isEmpty)
        #expect(sent.imageAttachments.count == 1)
        #expect(viewModel.pendingImage == nil)
    }

    @Test func sendDraftWithEmptyDraftAndNoPhotoSendsNothing() {
        let client = PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs())
        let viewModel = MessageView.ViewModel(client: client)
        viewModel.draftMessage = "   "

        viewModel.sendDraft()

        #expect(client.messages.isEmpty)
        #expect(viewModel.draftMessage == "   ")
    }

    @Test func visibleLiveMessagesSetBadgeInsteadOfScrollingWhenReadingHistory() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...2)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        viewModel.updateBottomVisibility(false)
        let previousScrollRequest = viewModel.scrollToBottomRequest

        viewModel.client.sendMessage("Live message")

        #expect(viewModel.displayedMessages.map(\.content).last == "Live message")
        #expect(viewModel.hasHiddenNewMessages)
        #expect(viewModel.scrollToBottomRequest == previousScrollRequest)
    }

    @Test func visibleLiveMessagesRequestScrollWhenAlreadyAtBottom() async {
        let database = configuredDatabase()
        saveNumberedMessages(in: database, ids: 1...2)
        let (viewModel, _) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        viewModel.updateBottomVisibility(true)
        let previousScrollRequest = viewModel.scrollToBottomRequest

        viewModel.client.sendMessage("Live message")

        #expect(viewModel.displayedMessages.map(\.content).last == "Live message")
        #expect(viewModel.hasHiddenNewMessages == false)
        #expect(viewModel.scrollToBottomRequest == previousScrollRequest + 1)
    }

    @Test func filteredLiveMessagesShowBadgeWithoutAppending() async {
        let database = configuredDatabase()
        let (viewModel, transport) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        viewModel.selectedMessageFilter = .chat
        await viewModel.waitForPaging()

        transport.emit(messageID: 20, content: "Schedule update", sourceHint: "Schedule")

        #expect(viewModel.displayedMessages.isEmpty)
        #expect(viewModel.hasHiddenNewMessages)
        #expect(database.containsMessage(serverID: 20))
    }

    @Test func clearingFiltersShowsHiddenNewMessages() async {
        let database = configuredDatabase()
        let (viewModel, transport) = makePagingViewModel(database: database, pageSize: 20)
        await viewModel.connect()
        viewModel.selectedMessageFilter = .chat
        await viewModel.waitForPaging()
        transport.emit(messageID: 20, content: "Schedule update", sourceHint: "Schedule")

        await viewModel.clearFiltersAndShowNewMessages()

        #expect(viewModel.selectedMessageFilter == .all)
        #expect(viewModel.hasHiddenNewMessages == false)
        #expect(viewModel.displayedMessages.map(\.content) == ["Schedule update"])
    }

    @Test func liveDockVisibilityReflectsSelectedFilter() {
        let viewModel = MessageView.ViewModel(client: PennyWebSocketClient(databaseService: configuredDatabase(), prefs: configuredPrefs()))

        #expect(viewModel.shouldShowLiveDock)

        viewModel.selectedMessageFilter = .chat
        #expect(viewModel.shouldShowLiveDock)

        viewModel.selectedMessageFilter = .penny
        #expect(viewModel.shouldShowLiveDock == false)

        viewModel.selectedMessageFilter = .collector
        #expect(viewModel.shouldShowLiveDock == false)
    }

    @Test func searchHistoryDefaultsToCompactAndLoadsSelectedFilter() async {
        let database = configuredDatabase()
        saveFilterFixture(in: database)
        let client = PennyWebSocketClient(databaseService: database, prefs: configuredPrefs())
        let viewModel = MessageSearchViewModel(client: client, pageSize: 2)

        viewModel.selectedFilter = .penny
        await viewModel.loadInitialHistory()

        #expect(viewModel.selectedLayout == .compact)
        #expect(viewModel.displayedMessages.map(\.id) == [2, 5])

        viewModel.selectedLayout = .media
        #expect(viewModel.selectedLayout == .media)
    }

    @Test func searchHistoryLoadsOlderPagesForTheSelectedFilter() async {
        let database = configuredDatabase()
        saveFilterFixture(in: database)
        let client = PennyWebSocketClient(databaseService: database, prefs: configuredPrefs())
        let viewModel = MessageSearchViewModel(client: client, pageSize: 2)

        viewModel.selectedFilter = .penny
        await viewModel.loadInitialHistory()
        await viewModel.loadMoreHistory()

        #expect(viewModel.displayedMessages.map(\.id) == [1, 2, 5])
    }

}

@MainActor
private final class MessageViewModelMockTransport: WebSocketTransport {
    private var onReceive: WebSocketTransport.ReceiveHandler?
    private var onFailure: WebSocketTransport.FailureHandler?
    var isConnected = false

    func connect(
        request: URLRequest,
        onReceive: @escaping WebSocketTransport.ReceiveHandler,
        onFailure: @escaping WebSocketTransport.FailureHandler
    ) {
        self.onReceive = onReceive
        self.onFailure = onFailure
        isConnected = true
    }

    func disconnect() {
        onReceive = nil
        onFailure = nil
        isConnected = false
    }

    func send(_ data: Data) async throws {}

    func emit(messageID: Int, content: String, sourceHint: String) {
        onReceive?(Data("""
        {
          "type": "messages",
          "messages": [
            {
              "id": \(messageID),
              "created_at": "2026-07-05T00:00:00Z",
              "content": "\(content)",
              "source_hint": "\(sourceHint)"
            }
          ]
        }
        """.utf8))
    }
}

@MainActor
private func makePagingViewModel(
    database: DatabaseService,
    pageSize: Int
) -> (MessageView.ViewModel, MessageViewModelMockTransport) {
    let transport = MessageViewModelMockTransport()
    let client = PennyWebSocketClient(
        databaseService: database,
        prefs: configuredPrefs(),
        webSocketClient: transport
    )
    return (MessageView.ViewModel(client: client, messagePageSize: pageSize), transport)
}

private func saveFilterFixture(in database: DatabaseService) {
    database.save(message: testMessage(id: 1, content: "Penny", sourceHint: "Penny"))
    database.save(message: testMessage(id: 2, content: "Startup", sourceHint: "Startup"))
    database.save(message: testMessage(id: 3, content: "Schedule", sourceHint: "Schedule"))
    database.save(message: testMessage(id: 4, content: "Chat", sourceHint: "Chat"))
    database.save(message: testMessage(id: 5, content: "Test Push", sourceHint: "Test Push"))
    database.save(message: testMessage(id: 6, content: "Notifier", sourceHint: "Notifier"))
    database.save(message: testMessage(id: 7, content: "Collector", sourceHint: "Collector: flight-deals"))
    database.save(message: testMessage(id: 8, serverID: nil, content: "Outgoing", isOutgoing: true))
}

private func saveNumberedMessages(in database: DatabaseService, ids: ClosedRange<Int>) {
    for id in ids {
        database.save(message: testMessage(id: id, content: "Message \(id)"))
    }
}

/// A solid-colour photo for the composer's pending-image tests (#3039).
@MainActor
private func solidTestImage() -> UIImage {
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let renderer = UIGraphicsImageRenderer(size: CGSize(width: 120, height: 80), format: format)
    return renderer.image { context in
        UIColor.systemTeal.setFill()
        context.fill(CGRect(x: 0, y: 0, width: 120, height: 80))
    }
}

private func testMessage(
    id: Int,
    serverID: Int? = nil,
    content: String,
    sourceHint: String? = nil,
    isOutgoing: Bool = false
) -> MessageModel {
    MessageModel(
        id: id,
        serverID: serverID ?? (isOutgoing ? nil : id),
        createdAt: Date(timeIntervalSince1970: TimeInterval(id)),
        content: content,
        sourceHint: sourceHint,
        imageAttachmentDataURLs: [],
        isOutgoing: isOutgoing
    )
}
