import SwiftUI
import PennyDesign
import PennyServices

/// The app shell (#3037): one centred glass tab capsule with a trailing
/// search circle floating over every screen — no traditional navigation bar,
/// no bottom tab bar. Four tabs host today's views; pushed pages keep the
/// capsule and add a glass back circle on the leading edge. Content scrolls
/// under the row and dissolves below it through the kit's `TopFade`.
struct AppShellView: View {
    @State private var selected: TopTab
    @State private var chatViewModel: MessageView.ViewModel
    @State private var browserModel: DataBrowserViewModel
    /// The top safe-area inset (the status bar), which the top fade holds
    /// opaque through.
    @State private var topInset: CGFloat = 0

    /// `chatViewModel` and `browserModel` are injectable so story mode
    /// (#2965) can render any tab from fixtures; the defaults are the real
    /// app's — one chat view model, and the browser model reading through
    /// its client.
    init(
        initialTab: TopTab = .chat,
        chatViewModel: MessageView.ViewModel? = nil,
        browserModel: DataBrowserViewModel? = nil
    ) {
        let resolvedChatViewModel = chatViewModel ?? MessageView.ViewModel()
        _selected = State(initialValue: initialTab)
        _chatViewModel = State(initialValue: resolvedChatViewModel)
        _browserModel = State(initialValue: browserModel ?? DataBrowserViewModel(runID: nil) {
            try await resolvedChatViewModel.client.dataBrowser.read($0)
        })
    }

    /// Where a tab's first content sits below the safe-area top: below the
    /// nav row and fully clear of the top fade (the fade is clear exactly
    /// `topFadeBleed` below the row).
    static var contentTopClearance: CGFloat {
        Tokens.chromeTopGap + Tokens.navControlHeight + Tokens.topFadeBleed
    }

    var body: some View {
        ZStack(alignment: .top) {
            tabContent
            topFade
            chromeRow
        }
        .background(Color(.systemGroupedBackground).ignoresSafeArea())
        .onGeometryChange(for: CGFloat.self, of: { $0.safeAreaInsets.top }, action: { topInset = $0 })
        .onAppear {
            syncBrowserTab(with: selected)
            browserModel.updateConnection(chatViewModel.client.canSend)
        }
        .onChange(of: selected) { _, newTab in
            syncBrowserTab(with: newTab)
            browserModel.contextChanged()
        }
        .onChange(of: chatViewModel.client.canSend) { _, connected in
            browserModel.updateConnection(connected)
        }
        .onChange(of: chatViewModel.client.browserMemoryRevision) { _, _ in
            if selected == .memory { browserModel.scheduleRefresh() }
        }
        .onChange(of: chatViewModel.client.browserRunRevision) { _, _ in
            if selected == .prompts || browserModel.selection != nil { browserModel.scheduleRefresh() }
        }
        .onDisappear { browserModel.stop() }
        .environment(\.shellContentTopClearance, Self.contentTopClearance)
        .environment(\.shellOpenPrompts) { runID in
            browserModel.linkedRunID = runID
            selected = .prompts
        }
    }

    // MARK: Tabs

    @ViewBuilder
    private var tabContent: some View {
        switch selected {
        case .chat:
            MessageView(viewModel: chatViewModel)
        case .memory:
            DataBrowserMemoriesView(model: browserModel, reconnect: chatViewModel.client.reconnect)
        case .prompts:
            DataBrowserPromptsView(model: browserModel, reconnect: chatViewModel.client.reconnect)
        case .settings:
            // The old Cancel/Save bar stays for now (the Settings redesign
            // ticket replaces it with save-as-you-change); padded below the
            // chrome row so the two rows of controls don't collide.
            SettingsView(client: chatViewModel.client)
                .padding(.top, Self.contentTopClearance)
        }
    }

    private func syncBrowserTab(with tab: TopTab) {
        switch tab {
        case .memory: browserModel.tab = .memories
        case .prompts: browserModel.tab = .prompts
        case .chat, .settings: break
        }
    }

    // MARK: Chrome

    /// The soft fade over the whole chrome zone: fully opaque from the
    /// screen's top through the nav row's bottom (status bar + gap + row),
    /// then fading out over `topFadeBleed` — a tab's first content at
    /// `contentTopClearance` sits exactly where the fade clears. Sized from
    /// the measured top safe-area inset, as the mock's `ChatCanvasTopFade`.
    private var topFade: some View {
        TopFade(
            background: Color(.systemGroupedBackground),
            height: topInset + Tokens.chromeTopGap + Tokens.navControlHeight + Tokens.topFadeBleed,
            opaqueThrough: topInset + Tokens.chromeTopGap + Tokens.navControlHeight
        )
        .frame(maxHeight: .infinity, alignment: .top)
        .ignoresSafeArea(edges: .top)
    }

    private var chromeRow: some View {
        ZStack {
            HStack {
                Spacer()
                TopTabBar(
                    selected: selected,
                    promptsBadge: chatViewModel.client.backgroundProgressRuns.count
                ) { ChatTabAvatar() }
                Spacer()
            }
            HStack {
                if showsBackButton {
                    GlassCircleButton(symbol: "chevron.left", onTap: goBack)
                        .accessibilityLabel("Back")
                }
                Spacer()
                // Per-tab search lands with #3045 and the later tickets;
                // until then the circle is inert.
                GlassCircle(symbol: "magnifyingglass")
            }
        }
        .padding(.horizontal, Tokens.margin)
        .padding(.top, Tokens.chromeTopGap)
    }

    /// The only pushed page today is the memory detail, reached when the
    /// browser model holds a selection.
    private var showsBackButton: Bool {
        selected == .memory && browserModel.selection != nil
    }

    private func goBack() {
        browserModel.selection = nil
    }
}

/// Penny's avatar for the chat tab — the app owns the `penny` asset; the
/// kit's `TopTabBar` sizes and badges whatever it is handed.
struct ChatTabAvatar: View {
    var body: some View {
        Image("penny")
            .resizable()
            .scaledToFill()
            .clipShape(Circle())
    }
}

// MARK: Environment

/// How far a tab's scrolling content starts below the safe-area top: the
/// nav row plus the top fade, so content begins fully clear of both and
/// scrolls up under them.
private struct ShellContentTopClearanceKey: EnvironmentKey {
    static let defaultValue: CGFloat = 0
}

extension EnvironmentValues {
    var shellContentTopClearance: CGFloat {
        get { self[ShellContentTopClearanceKey.self] }
        set { self[ShellContentTopClearanceKey.self] = newValue }
    }
}

/// Opens the Prompts tab on a run — the chat long-press menu's "View
/// prompts" route, replacing the data-browser popover.
private struct ShellOpenPromptsKey: EnvironmentKey {
    static let defaultValue: (String) -> Void = { _ in }
}

extension EnvironmentValues {
    var shellOpenPrompts: (String) -> Void {
        get { self[ShellOpenPromptsKey.self] }
        set { self[ShellOpenPromptsKey.self] = newValue }
    }
}

#Preview {
    AppShellView()
}
