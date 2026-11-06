import PennyServices

/// Queued-message bookkeeping (#3043) for `MessageView.ViewModel`: a
/// message sent while Penny is working is sent immediately, same as ever
/// (the server's turn inbox, #2985, is what actually holds and injects
/// it) — this only tracks which already-sent ids still show the queued
/// bubble style, and clears that mark the moment the run shows the
/// server picked the message up.
extension MessageView.ViewModel {
    /// Marks a just-sent message as queued-looking, recording the run's
    /// step count at the moment it was sent (`sendDraft()`) — the
    /// baseline `clearQueuedMarksOnStepAdvance()` compares against.
    func markMessageAsQueued(_ messageID: Int, runStepCountAtSend: Int) {
        queuedMessageIDs.insert(messageID)
        queuedMessageStepBaselines[messageID] = runStepCountAtSend
    }

    /// Whether a message still shows the queued bubble style.
    func isMessageQueued(_ messageID: Int) -> Bool {
        queuedMessageIDs.contains(messageID)
    }

    /// Clears a queued mark once the run has moved past the step it was
    /// sent in (#2985: a later `step_started` means the server has
    /// folded the message into the running turn) — called from
    /// `syncDock()` on every progress update.
    func clearQueuedMarksOnStepAdvance() {
        guard !queuedMessageIDs.isEmpty, let currentStepCount = dockRun?.steps.count else { return }
        let advancedIDs = queuedMessageIDs.filter { (queuedMessageStepBaselines[$0] ?? 0) < currentStepCount }
        for id in advancedIDs {
            queuedMessageIDs.remove(id)
            queuedMessageStepBaselines.removeValue(forKey: id)
        }
    }

    /// Clears every remaining queued mark when the run ends (#2985: the
    /// message already reached Penny's turn one way or another by then,
    /// whether or not a further `step_started` ever showed it) — called
    /// wherever the dock detects the foreground run is over.
    func clearQueuedMarksOnRunEnd() {
        queuedMessageIDs.removeAll()
        queuedMessageStepBaselines.removeAll()
    }
}
