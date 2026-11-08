import Observation
import SwiftUI
import PennyDesign
import PennyServices

extension MessageView {
    /// The live dock's lifecycle in the conversation's flow (#3042, #3130):
    /// hidden, showing a run's live stream, or resolving — the run just
    /// ended (a reply arrived, or its grace window ran out) and the dock is
    /// on its way out, playing its own stock exit transition. When a reply
    /// matched, it stays out of the grid until the dock's exit genuinely
    /// finishes, then gets its OWN separate stock arrival transition — two
    /// sequential stock animations, never one faked into morphing into the
    /// other.
    enum DockPhase: Equatable {
        case hidden
        case working
        case resolving
    }

    enum MessageLayout: Int, CaseIterable, Identifiable, Sendable {
        case message = 1
        case compact = 2
        case media = 3

        var id: Self { self }

        var title: String {
            switch self {
            case .message:
                return "Message"
            case .compact:
                return "Compact"
            case .media:
                return "Media"
            }
        }

        var systemImage: String {
            "\(rawValue).circle"
        }
    }

    enum MessageFilter: String, CaseIterable, Identifiable, Sendable {
        case all
        case penny
        case chat
        case notifier
        case collector

        var id: Self { self }

        var title: String {
            switch self {
            case .all:
                return "All Messages"
            case .penny:
                return "Penny"
            case .chat:
                return "Chat"
            case .notifier:
                return "Notifier"
            case .collector:
                return "Collector"
            }
        }

        var systemImage: String {
            switch self {
            case .all:
                return "tray.full"
            case .penny:
                return "sparkles"
            case .chat:
                return "bubble.left.and.bubble.right"
            case .notifier:
                return "bell"
            case .collector:
                return "tray.and.arrow.down"
            }
        }

        var pageFilter: MessagePageFilter {
            switch self {
            case .all:
                return .all
            case .penny:
                return .penny
            case .chat:
                return .chat
            case .notifier:
                return .notifier
            case .collector:
                return .collector
            }
        }
    }

    @MainActor
    @Observable
    final class ViewModel {
        struct ComposerPresentation {
            let placeholder: String
            let tint: Color?
        }

        var client = PennyWebSocketClient()
        var draftMessage = ""
        /// The photo picked in the composer, waiting to ride the next send —
        /// shown as the field's thumbnail, removed by its glass ✕.
        var pendingImage: UIImage?
        var hasHiddenNewMessages = false
        var replyMessage: ChatMessage?
        var selectedMessageLayout: MessageLayout = .message
        var selectedMessageFilter: MessageFilter = .all {
            didSet {
                guard selectedMessageFilter != oldValue else { return }
                if selectedMessageFilter == .all {
                    hasHiddenNewMessages = false
                }
                client.updateLiveMessageFilter(selectedMessageFilter.pageFilter)
                reloadMessagesForSelectedFilter()
            }
        }
        var displayedMessages: [ChatMessage] = [] {
            didSet {
                handleDisplayedMessagesChanged(previousMessages: oldValue)
            }
        }
        var isAtBottom = true
        var isLoadingOlderMessages = false
        var hasMoreOlderMessages = false
        var scrollToBottomRequest = 0
        var shouldSettleScrollToBottom = true
        /// Find-in-chat's state (#3045) — see `MessageView+Search.swift`.
        /// `var`, not `let`: the chrome row's `$chatViewModel.search.query`
        /// binding needs a writable key path to reach through, even though
        /// `search` itself (a reference type) is never reassigned.
        var search = ChatSearchState()

        // MARK: Live dock (#3042)
        var dockPhase: DockPhase = .hidden
        /// The foreground run the dock was showing — kept set through
        /// `.resolving` (cleared only by `dockExitFinished()`) purely so a
        /// new run arriving before that fires can tell it isn't this one
        /// (`syncDock`'s stuck-resolving recovery below); the dock view
        /// itself is already gone the instant `dockPhase` leaves `.working`
        /// (#3130) — nothing renders `dockRun` during `.resolving` anymore.
        var dockRun: AgentProgressRunItem?
        /// The id of the just-arrived reply held out of the grid while
        /// `dockPhase == .resolving` with a match (#3130) — revealed, with
        /// its own stock arrival transition, only once `dockExitFinished()`
        /// runs. `nil` during the plain fade-out of a run that ended with
        /// nothing to show for it.
        var resolvingMessageID: Int?
        /// A permission request the dock itself just answered (#3044,
        /// `MessageView+Permission.swift`) — lives only as long as this
        /// run's dock, cleared below with the rest of its state.
        var resolvedPermissionItem: LiveDockItem?

        // MARK: Queued messages (#3043)
        /// Ids of messages sent while `dockPhase == .working` — the
        /// server's turn inbox (#2985) holds a message that arrives
        /// mid-turn and injects it into the running turn after its
        /// current tool step, rather than starting a second turn, so the
        /// message is sent immediately (never held back here) and only
        /// shown in the queued bubble style until that injection shows up
        /// as the run's next step. Tracked by id only — nothing persisted,
        /// since after a relaunch it is simply a sent message.
        var queuedMessageIDs: Set<Int> = []
        /// Each queued id's run step count at the moment it was sent
        /// (`MessageView+Queue.swift`) — the mark clears once the run's
        /// step count exceeds this, meaning a `step_started` after the
        /// send arrived (the server picked the message up).
        @ObservationIgnored var queuedMessageStepBaselines: [Int: Int] = [:]

        @ObservationIgnored private var dockPendingTask: Task<Void, Never>?
        @ObservationIgnored private let messagePageSize: Int
        @ObservationIgnored private var nextOlderCursor: MessagePageCursor?
        @ObservationIgnored private var pagingTask: Task<Void, Never>?
        @ObservationIgnored private var suppressDisplayedMessageChanges = false
        @ObservationIgnored private var hasBoundLiveMessages = false
        @ObservationIgnored private var olderPagingEnabled = false
        @ObservationIgnored private var messagePagingGeneration = 0
        @ObservationIgnored private var reservedOlderMessageLoad: OlderMessageLoad?
        @ObservationIgnored private var isRestoringOlderMessageScroll = false

        init(client: PennyWebSocketClient? = nil, messagePageSize: Int = 30) {
            let resolvedClient = client ?? PennyWebSocketClient()
            self.client = resolvedClient
            self.messagePageSize = max(1, messagePageSize)
            search.attach(resolvedClient)
        }

        /// The live dock (#3042) only shows on the filters that read as
        /// "chat" — the same filters the old typing row showed on.
        var shouldShowLiveDock: Bool {
            selectedMessageFilter == .all || selectedMessageFilter == .chat
        }

        /// A cheap, Equatable proxy for "the foreground run's content
        /// changed": `AgentProgressRunItem` isn't Equatable, so the view's
        /// `onChange` can't watch it directly. Grows while a run is live;
        /// -1 once it's gone — the only transition `syncDock` acts on.
        var foregroundProgressVersion: Int {
            guard let run = client.foregroundProgress else { return -1 }
            return run.steps.reduce(0) { $0 + 1 + $1.tools.count }
        }

        /// Keeps `dockRun` current while a foreground run is live, and
        /// reacts the moment it disappears (run_finished). Tracks the run
        /// regardless of `shouldShowLiveDock` — the view already gates
        /// *rendering* on that filter separately, so this can stay simple:
        /// state always mirrors the live run, never a value that could go
        /// stale while a non-chat filter is selected and then reappear
        /// wrong the moment the filter switches back.
        ///
        /// Resets to `.working` whenever a run is live and the dock isn't
        /// already `.working` — covering not just `.hidden` but a run that
        /// starts while the dock is still `.resolving` a previous one (an
        /// earlier reply's reveal never finished because this new run's own
        /// steps started arriving first): that run's content must not sit
        /// behind a resolve that's still pending. The other safety net for a
        /// stuck `.resolving` — no new run, but the view that would have run
        /// the exit animation's completion was never on screen to do it —
        /// is `finishLeftoverResolveOnAppear()` below, NOT this function:
        /// run-end can land here while a matched reply's exit is still
        /// genuinely in flight (the two arrive close together, either
        /// order), and finishing the dock early from here would cut that
        /// exit short and pop the reply in before its own transition
        /// completes — exactly the sequencing this issue exists to fix.
        func syncDock() {
            guard let run = client.foregroundProgress else {
                handleForegroundRunEnded()
                return
            }
            dockPendingTask?.cancel()
            dockRun = run
            clearQueuedMarksOnStepAdvance()
            if dockPhase != .working {
                dockPhase = .working
            }
        }

        /// The run just left `client.agentProgressRuns` with the dock still
        /// showing it. Only acts while `dockPhase == .working`: a reply
        /// landing just before `run_finished` already moved `dockPhase` to
        /// `.resolving` via `beginResolvingIfMatched`, and `run_finished`
        /// landing a beat later must leave that exit alone — cutting it
        /// short here would pop the reply in before its own transition
        /// finishes, the very race this guard exists to avoid (#3130). The
        /// stuck-`.resolving` case this does NOT cover — no new run ever
        /// starts, and the view that would run the exit animation's
        /// completion was never on screen to fire it — is
        /// `finishLeftoverResolveOnAppear()` below. Otherwise start the
        /// grace window (#3042): either a reply lands before it elapses and
        /// its own exit starts, or the run ended with nothing to show for
        /// it and the dock just leaves.
        private func handleForegroundRunEnded() {
            clearQueuedMarksOnRunEnd()
            guard dockPhase == .working, dockRun != nil else { return }
            dockPendingTask?.cancel()
            dockPendingTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(Tokens.dockResolveGraceSeconds))
                guard let self, !Task.isCancelled, self.dockPhase == .working else { return }
                // This is the dock just leaving with nothing to reveal, so
                // it's animated here explicitly — unlike the later,
                // purely-bookkeeping `dockExitFinished()` after a matched
                // reply's reveal (the view has nothing left to visibly
                // settle by then).
                withAnimation(.easeInOut(duration: Tokens.dockExitDuration)) {
                    self.dockExitFinished()
                }
            }
        }

        /// A newly arrived reply carrying the dock's own run id starts the
        /// dock's real exit now (#3130) — never waiting for `run_finished`,
        /// which can trail a message that has already arrived. Returns
        /// whether a match was found, so the view's `withAnimation(...)
        /// completion:` knows whether it has anything to reveal once the
        /// exit genuinely finishes — never a fixed-duration sleep guessing
        /// when that animation is done.
        @discardableResult
        func beginResolvingIfMatched(appendedMessages: [ChatMessage]) -> Bool {
            guard let dockRun, dockPhase != .hidden else { return false }
            guard let reply = appendedMessages.first(where: { $0.runID == dockRun.id && !$0.isOutgoing }) else { return false }
            clearQueuedMarksOnRunEnd()
            dockPendingTask?.cancel()
            resolvingMessageID = reply.id
            dockPhase = .resolving
            return true
        }

        /// The dock clears on completion (#3042, #3130): hidden, with
        /// nothing left to reveal or fade out — called once the grace
        /// window elapses with no reply, once a matched reply's exit
        /// animation completion fires, or by the stuck-`.resolving`
        /// fallback `finishLeftoverResolveOnAppear()` below. Unconditional:
        /// a caller that already knows the state changed (a direct test,
        /// the grace-window path) relies on this always resetting fully.
        ///
        /// A matched reply (`resolvingMessageID` was set) was excluded
        /// from the grid the whole time the dock played its exit — it
        /// becomes the conversation's new last item only on THIS call, so
        /// this is also where that arrival gets the same "new content"
        /// handling `handleDisplayedMessagesChanged` gives a live message:
        /// scroll to it if already at the bottom, or mark it hidden for
        /// the "show new messages" pill otherwise. Without this, nothing
        /// ever re-anchors the scroll once the dock's space is replaced by
        /// the reply, and it lands wherever the dock happened to leave the
        /// viewport — partially or fully off screen (#3130).
        func dockExitFinished() {
            let revealedAReply = resolvingMessageID != nil
            dockPhase = .hidden
            dockRun = nil
            resolvingMessageID = nil
            resolvedPermissionItem = nil
            guard revealedAReply else { return }
            if isAtBottom {
                scrollToBottomRequest += 1
            } else {
                hasHiddenNewMessages = true
            }
        }

        /// The stuck-`.resolving` safety net the chat view's own `.onAppear`
        /// calls, alongside `syncDock()`, every time it (re)appears (#3130):
        /// a matched reply's exit animation reports its completion through
        /// `withAnimation(...) completion:` in `MessageView
        /// .prepareArrivalAnimations`, but that only fires while some view
        /// is actually running the animation — if the chat tab was
        /// backgrounded or a different tab was selected when it would have
        /// completed, nothing ever called `dockExitFinished()`, and the
        /// reply stays excluded from the grid until this runs. No
        /// animation: the dock itself already left the hierarchy while
        /// this view was away, so there's nothing left to visibly settle.
        /// `syncDock()`'s own next-run recovery (resetting straight to
        /// `.working`) still covers the case where a NEW run starts first.
        func finishLeftoverResolveOnAppear() {
            guard dockPhase == .resolving else { return }
            dockExitFinished()
        }

        var composerPresentation: ComposerPresentation {
            switch client.conversationState {
            case .elicit:
                ComposerPresentation(
                    placeholder: "Teach Penny",
                    tint: Color.purple.opacity(0.18)
                )
            case .learn:
                ComposerPresentation(
                    placeholder: "Approve or correct",
                    tint: Color.blue.opacity(0.16)
                )
            case .request:
                ComposerPresentation(
                    placeholder: "Add the missing detail",
                    tint: Color.orange.opacity(0.18)
                )
            case .idle, .apply, nil:
                ComposerPresentation(placeholder: "Message", tint: nil)
            }
        }

        var canLoadOlderMessages: Bool {
            olderPagingEnabled && hasMoreOlderMessages && !isLoadingOlderMessages && !isRestoringOlderMessageScroll
        }

        func connect() async {
            bindLiveMessagesIfNeeded()
            await loadLatestMessages()
            await client.connect()
        }

        func disconnect() {
            client.unbindLiveMessages()
            hasBoundLiveMessages = false
            client.disconnect()
        }

        func reconnect() {
            bindLiveMessagesIfNeeded()
            client.reconnect()
        }

        func loadLatestMessages() async {
            messagePagingGeneration += 1
            reservedOlderMessageLoad = nil
            isRestoringOlderMessageScroll = false
            isLoadingOlderMessages = false
            olderPagingEnabled = false
            let page = await client.requestMessagePage(MessagePageRequest(limit: messagePageSize, filter: selectedMessageFilter.pageFilter))
            replaceDisplayedMessages(with: page.messages)
            nextOlderCursor = page.nextCursor
            hasMoreOlderMessages = page.hasMore
            scrollToBottomRequest += 1
        }

        func reserveOlderMessageLoad() -> Int? {
            guard canLoadOlderMessages else { return nil }
            guard let nextOlderCursor, let anchorID = displayedMessages.first?.id else { return nil }
            isLoadingOlderMessages = true
            reservedOlderMessageLoad = OlderMessageLoad(
                cursor: nextOlderCursor,
                filter: selectedMessageFilter.pageFilter,
                generation: messagePagingGeneration
            )
            return anchorID
        }

        func loadReservedOlderMessages() async -> Bool {
            guard let reservation = reservedOlderMessageLoad else {
                isLoadingOlderMessages = false
                return false
            }
            defer {
                isLoadingOlderMessages = false
                reservedOlderMessageLoad = nil
            }

            try? await Task.sleep(for: .milliseconds(250))
            guard messagePagingGeneration == reservation.generation,
                  nextOlderCursor == reservation.cursor,
                  selectedMessageFilter.pageFilter == reservation.filter else {
                return false
            }

            let page = await client.requestMessagePage(
                MessagePageRequest(
                    limit: messagePageSize,
                    before: reservation.cursor,
                    filter: reservation.filter
                )
            )
            self.nextOlderCursor = page.nextCursor
            hasMoreOlderMessages = page.hasMore

            let existingIDs = Set(displayedMessages.map(\.id))
            let olderMessages = page.messages.filter { !existingIDs.contains($0.id) }
            guard !olderMessages.isEmpty else { return false }

            suppressDisplayedMessageChanges = true
            displayedMessages.insert(contentsOf: olderMessages, at: 0)
            suppressDisplayedMessageChanges = false
            isRestoringOlderMessageScroll = true
            return true
        }

        func finishOlderMessageScrollRestoration() {
            isRestoringOlderMessageScroll = false
        }

        func enableOlderPaging() {
            olderPagingEnabled = true
        }

        func requestScrollToBottom(shouldSettleLayout: Bool = true) {
            olderPagingEnabled = false
            reservedOlderMessageLoad = nil
            isRestoringOlderMessageScroll = false
            isAtBottom = true
            shouldSettleScrollToBottom = shouldSettleLayout
            scrollToBottomRequest += 1
        }

        @discardableResult
        func prepareComposerFocus() -> Bool {
            guard selectedMessageLayout != .message else { return false }
            selectedMessageLayout = .message
            return true
        }

        func updateBottomVisibility(_ isVisible: Bool) {
            isAtBottom = isVisible
            if isVisible && selectedMessageFilter == .all {
                hasHiddenNewMessages = false
            }
        }

        func waitForPaging() async {
            await pagingTask?.value
        }

        func waitForFiltering() async {
            await waitForPaging()
        }

        func startReply(to message: ChatMessage) {
            replyMessage = message
        }

        func cancelReply() {
            replyMessage = nil
        }

        func replySummary(for message: ChatMessage) -> String {
            let normalized = message.content
                .split(whereSeparator: \.isNewline)
                .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            return normalized.isEmpty ? "Attachment" : normalized
        }

        func clearFiltersAndShowNewMessages() async {
            hasHiddenNewMessages = false
            if selectedMessageFilter == .all {
                await loadLatestMessages()
            } else {
                selectedMessageFilter = .all
                await waitForPaging()
            }
        }

        func sendDraft() {
            let trimmedMessage = draftMessage.trimmingCharacters(in: .whitespacesAndNewlines)
            let pendingPhoto = pendingImage
            guard !trimmedMessage.isEmpty || pendingPhoto != nil else { return }

            let messageToSend = replyMessage.map { replyContent(for: trimmedMessage, replyingTo: $0) } ?? trimmedMessage
            draftMessage = ""
            pendingImage = nil
            replyMessage = nil
            if selectedMessageFilter != .all {
                selectedMessageFilter = .all
            }
            // Always sent immediately (#3043, #2985): a message arriving
            // mid-turn joins the server's turn inbox and is injected into
            // the running turn, never queued client-side. The sent id is
            // marked for the queued bubble style while the dock is busy —
            // `dockPhase`/`dockRun` are read BEFORE the send so a run that
            // ends in the same synchronous call never leaves a stale mark.
            let wasWorking = dockPhase == .working
            let runStepCountAtSend = dockRun?.steps.count ?? 0
            let sentMessageID = client.sendMessage(messageToSend, images: pendingPhoto.map { [$0] })
            if wasWorking {
                markMessageAsQueued(sentMessageID, runStepCountAtSend: runStepCountAtSend)
            }
            requestScrollToBottom()
        }

        func handleScenePhaseChange(_ phase: ScenePhase) {
            switch phase {
            case .active:
                Task { await connect() }
            case .background:
                disconnect()
            case .inactive:
                break
            @unknown default:
                break
            }
        }

        private func replyContent(for message: String, replyingTo originalMessage: ChatMessage) -> String {
            """
            \(message)

            In reply to \(originalMessage.isOutgoing ? "you" : (originalMessage.sourceHint ?? "Penny")):
            > \(replySummary(for: originalMessage))
            """
        }

        private func reloadMessagesForSelectedFilter() {
            pagingTask?.cancel()
            pagingTask = Task { [weak self] in
                await self?.loadLatestMessages()
            }
        }

        private func bindLiveMessagesIfNeeded() {
            if hasBoundLiveMessages {
                client.updateLiveMessageFilter(selectedMessageFilter.pageFilter)
                return
            }

            client.bindLiveMessages(
                Binding(
                    get: { [weak self] in self?.displayedMessages ?? [] },
                    set: { [weak self] messages in self?.displayedMessages = messages }
                ),
                hasNewMessages: Binding(
                    get: { [weak self] in self?.hasHiddenNewMessages ?? false },
                    set: { [weak self] hasNewMessages in self?.hasHiddenNewMessages = hasNewMessages }
                ),
                filter: selectedMessageFilter.pageFilter,
                historicalMessages: { [weak self] messages in
                    self?.mergeNewestHistoricalMessages(messages)
                }
            )
            hasBoundLiveMessages = true
        }

        private func mergeNewestHistoricalMessages(_ incomingMessages: [ChatMessage]) {
            guard !incomingMessages.isEmpty else { return }

            let existingIDs = Set(displayedMessages.map(\.id))
            let candidates = incomingMessages.filter { !existingIDs.contains($0.id) }
            guard !candidates.isEmpty else { return }

            guard let newest = displayedMessages.last else {
                replaceDisplayedMessages(with: candidates.sorted { $0.createdAt < $1.createdAt })
                scrollToBottomRequest += 1
                return
            }

            let newestMessages = candidates.filter {
                $0.createdAt > newest.createdAt || ($0.createdAt == newest.createdAt && $0.id > newest.id)
            }
            guard !newestMessages.isEmpty else { return }

            suppressDisplayedMessageChanges = true
            displayedMessages.append(contentsOf: newestMessages)
            displayedMessages.sort {
                $0.createdAt < $1.createdAt || ($0.createdAt == $1.createdAt && $0.id < $1.id)
            }
            suppressDisplayedMessageChanges = false
            scrollToBottomRequest += 1
        }

        private func replaceDisplayedMessages(with messages: [ChatMessage]) {
            suppressDisplayedMessageChanges = true
            displayedMessages = messages
            suppressDisplayedMessageChanges = false
        }

        private func handleDisplayedMessagesChanged(previousMessages: [ChatMessage]) {
            guard !suppressDisplayedMessageChanges else { return }
            guard displayedMessages.count > previousMessages.count else { return }

            if isAtBottom {
                scrollToBottomRequest += 1
            } else {
                hasHiddenNewMessages = true
            }
        }
    }

    private struct OlderMessageLoad {
        let cursor: MessagePageCursor
        let filter: MessagePageFilter
        let generation: Int
    }
}
