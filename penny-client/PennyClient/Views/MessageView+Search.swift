import Foundation
import Observation
import PennyServices

/// Find-in-chat's search state (#3045, wired to the stock system search
/// field in #3111/#3129): the keyword that highlights matches in the chat
/// below and scrolls to the newest one. Matches come from
/// `PennyService.searchMessages` — the whole local history, not just what
/// the chat view currently has loaded — so the match order is never a
/// function of how far the view has paged. No counter or stepping between
/// matches (the code owner approved highlights only) — the system search
/// field has no room for them, and nothing here drives one anymore.
@MainActor
@Observable
final class ChatSearchState {
    var isSearching = false
    /// Re-searches on every change — a plain local `LIKE` query, cheap
    /// enough per keystroke that debouncing isn't worth the complexity.
    var query = "" {
        didSet {
            guard query != oldValue else { return }
            refresh()
        }
    }
    private(set) var matches: [MessageKeywordMatch] = []
    private(set) var currentIndex: Int?

    @ObservationIgnored private var client: PennyService?

    /// Bound once, from the owning `MessageView.ViewModel`'s `init` — the
    /// client the search queries never changes for the view model's
    /// lifetime (reconnects reuse the same instance).
    func attach(_ client: PennyService) {
        self.client = client
    }

    var currentMatchID: Int? {
        guard let currentIndex, matches.indices.contains(currentIndex) else { return nil }
        return matches[currentIndex].id
    }

    /// The query to highlight in the chat, or `nil` when there's nothing
    /// to highlight (closed, or an empty field).
    var highlightQuery: String? {
        guard isSearching, !trimmedQuery.isEmpty else { return nil }
        return trimmedQuery
    }

    func open() {
        isSearching = true
    }

    func close() {
        isSearching = false
        query = ""
    }

    private var trimmedQuery: String {
        query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func refresh() {
        guard !trimmedQuery.isEmpty, let client else {
            matches = []
            currentIndex = nil
            return
        }
        matches = client.searchMessages(trimmedQuery, filter: .all)
        currentIndex = matches.isEmpty ? nil : 0
    }
}

extension MessageView.ViewModel {
    /// Whether `id` is already in `displayedMessages` — find-in-chat pages
    /// further back when a match isn't (#3045).
    func isSearchMatchLoaded(_ id: Int) -> Bool {
        displayedMessages.contains { $0.id == id }
    }

    /// Pages older history into `displayedMessages` until `id` is present,
    /// or there's nothing more to load. Reuses the same reservation the
    /// scroll-triggered older-load uses (`reserveOlderMessageLoad` /
    /// `loadReservedOlderMessages`), so the two never load concurrently —
    /// a reservation already in flight simply fails this one, which stops
    /// rather than double-loading.
    @discardableResult
    func loadUntilSearchMatchIsPresent(_ id: Int) async -> Bool {
        while !isSearchMatchLoaded(id) {
            guard reserveOlderMessageLoad() != nil else { return false }
            let loadedMore = await loadReservedOlderMessages()
            finishOlderMessageScrollRestoration()
            guard loadedMore else { return false }
        }
        return true
    }
}
