import PennyDesign
import PennyServices

/// The live dock's permission ask (#3044): the pending domain-read request
/// the server already sends (`client.permissionPrompt`, today shown only in
/// Settings → Permission Request) appears as the dock's trailing row while a
/// foreground run is live, with Deny and Allow. A request that arrives with
/// no live run never shows here — `dockPermission` is `nil` whenever
/// `dockPhase != .working` — and stays exactly where it is today, in
/// Settings.
///
/// Answering FROM THE DOCK turns the row into a normal stream item
/// ("Allowed"/"Denied" + the domain, `.tool` kind, done) via
/// `resolvedPermissionItem`. Answering from Settings instead clears
/// `client.permissionPrompt` the same way (both read the one shared
/// `PennyService` instance, `AppShellView.swift`), so the dock's pending row
/// disappears there too — but the dock never renders an outcome row for it:
/// the server's `permission_dismiss` frame carries only the request id, no
/// allowed/denied value (`penny/channels/ios/channel.py`,
/// `handle_permission_dismiss` — the answer rides the separate, unrelated
/// `domain_permissions_sync` broadcast instead), so the dock only knows the
/// outcome when it made the call itself.
extension MessageView.ViewModel {
    var dockPermission: LiveDockPermission? {
        guard dockPhase == .working, let prompt = client.permissionPrompt else { return nil }
        return LiveDockPermission(
            domain: prompt.domain,
            reason: "To finish your question",
            onAllow: { [weak self] in self?.decidePermission(allowed: true) },
            onDeny: { [weak self] in self?.decidePermission(allowed: false) }
        )
    }

    /// The dock's stream (#3042) plus, once answered from here, the
    /// resolved permission row trailing it.
    var dockStreamItems: [LiveDockItem] {
        guard let resolvedPermissionItem else { return dockRun?.liveDockItems ?? [] }
        return (dockRun?.liveDockItems ?? []) + [resolvedPermissionItem]
    }

    func decidePermission(allowed: Bool) {
        guard let prompt = client.permissionPrompt else { return }
        client.decidePermission(requestID: prompt.requestID, allowed: allowed)
        resolvedPermissionItem = LiveDockItem(
            id: "permission-\(prompt.requestID)",
            kind: .tool(
                symbol: "lock.fill",
                label: "\(allowed ? "Allowed" : "Denied") \(prompt.domain)",
                running: false
            )
        )
    }
}
