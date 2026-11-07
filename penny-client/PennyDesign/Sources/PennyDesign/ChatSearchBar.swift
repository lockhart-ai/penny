import SwiftUI

/// Find-in-chat's in-place search bar (#3045): the chrome row's tab
/// capsule and search circle replaced — on the same row and centre line —
/// by a row of separate glass pieces, the way the normal row is a tab
/// capsule and a search circle, not one capsule containing everything: the
/// glass field capsule (magnifier, query, the match counter at its
/// trailing end) filling the remaining width, two glass circles stepping
/// between matches, and a glass circle closing the search. 8pt between the
/// pieces, edges on the margins. The field is the tab capsule's
/// construction (content row + 4pt glass padding each side, to
/// `Tokens.navControlHeight`); the circles are the app's one glass circle
/// (`GlassCircleButton`) — every glass control in the row reads 44pt.
public struct ChatSearchBar: View {
    @Binding private var query: String
    private let matchCountText: String?
    private let arrowsEnabled: Bool
    private let isFocused: FocusState<Bool>.Binding
    private let onStepToOlderMatch: () -> Void
    private let onStepToNewerMatch: () -> Void
    private let onClose: () -> Void

    public init(
        query: Binding<String>,
        matchCountText: String?,
        arrowsEnabled: Bool,
        isFocused: FocusState<Bool>.Binding,
        onStepToOlderMatch: @escaping () -> Void,
        onStepToNewerMatch: @escaping () -> Void,
        onClose: @escaping () -> Void
    ) {
        _query = query
        self.matchCountText = matchCountText
        self.arrowsEnabled = arrowsEnabled
        self.isFocused = isFocused
        self.onStepToOlderMatch = onStepToOlderMatch
        self.onStepToNewerMatch = onStepToNewerMatch
        self.onClose = onClose
    }

    public var body: some View {
        HStack(spacing: 8) {
            field
            GlassCircleButton(symbol: "chevron.up", dimmed: !arrowsEnabled, onTap: onStepToOlderMatch)
                .accessibilityLabel("Previous match")
                .accessibilityIdentifier(AXID.searchOlderButton)
            GlassCircleButton(symbol: "chevron.down", dimmed: !arrowsEnabled, onTap: onStepToNewerMatch)
                .accessibilityLabel("Next match")
                .accessibilityIdentifier(AXID.searchNewerButton)
            GlassCircleButton(symbol: "xmark", onTap: onClose)
                .accessibilityLabel("Close search")
                .accessibilityIdentifier(AXID.searchCloseButton)
        }
        .padding(.horizontal, Tokens.margin)
    }

    /// The field capsule: the magnifier, the query (or its placeholder),
    /// and the match counter at the trailing end. Same 44pt outer height as
    /// the tab capsule it replaces.
    private var field: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.secondary)
            TextField("Search messages", text: $query)
                .focused(isFocused)
                .font(.body)
                .textFieldStyle(.plain)
                .submitLabel(.search)
                .autocorrectionDisabled()
                .accessibilityIdentifier(AXID.searchField)
            if let matchCountText {
                Text(matchCountText)
                    .font(.subheadline.weight(.medium).monospacedDigit())
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier(AXID.searchCounter)
            }
        }
        .padding(.leading, 14)
        .padding(.trailing, 12)
        // Content row of the 44pt field capsule: minus the glass's 4pt inset each side.
        .frame(height: Tokens.navControlHeight - 8)
        .padding(.vertical, 4)
        .glassEffect(.regular, in: .capsule)
    }
}
