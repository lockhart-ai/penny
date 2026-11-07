import SwiftUI
import PennyDesign
import PennyServices

/// The app shell (#3111): one stock `NavigationStack` with a visible system
/// bar. A leading glass page `Menu` switches between Chat, Memory, Prompts
/// and Settings under the same bar (no back button on a page switch);
/// Penny's avatar and name sit centred (`.principal`); search is trailing
/// (`DefaultToolbarItem(kind: .search)`), each page supplying its own
/// `.searchable`. Drill-downs (today, only a memory's detail) push onto the
/// stack via a typed `ShellRoute` and return with the system back button.
/// Replaces the hand-built top tab capsule and glass circles, whose taps
/// were swallowed by the hidden navigation bars beneath them (#3111).
struct AppShellView: View {
    @State private var page: TopTab
    @State private var chatViewModel: MessageView.ViewModel
    @State private var browserModel: DataBrowserViewModel

    /// Story mode only (#3042): passed straight through to the chat tab's
    /// `MessageView`, freezing the live dock mid-morph.
    private let dockResolveProgress: Double?

    /// `chatViewModel` and `browserModel` are injectable so story mode
    /// (#2965) can render any page from fixtures; the defaults are the real
    /// app's — one chat view model, and the browser model reading through
    /// its client.
    init(
        initialTab: TopTab = .chat,
        chatViewModel: MessageView.ViewModel? = nil,
        browserModel: DataBrowserViewModel? = nil,
        dockResolveProgress: Double? = nil
    ) {
        let resolvedChatViewModel = chatViewModel ?? MessageView.ViewModel()
        _page = State(initialValue: initialTab)
        _chatViewModel = State(initialValue: resolvedChatViewModel)
        _browserModel = State(initialValue: browserModel ?? DataBrowserViewModel(runID: nil) {
            try await resolvedChatViewModel.client.dataBrowser.read($0)
        })
        self.dockResolveProgress = dockResolveProgress
    }

    /// The one drill-down destination the shell pushes today — a typed
    /// route, not a raw `String`, so a second destination added later
    /// can't compile without a matching `navigationDestination` arm.
    enum ShellRoute: Hashable {
        case memoryDetail(String)
    }

    var body: some View {
        NavigationStack(path: routePath) {
            content(for: page)
                .toolbar { pageToolbar }
                .navigationDestination(for: ShellRoute.self) { route in
                    switch route {
                    case .memoryDetail(let name):
                        DataBrowserMemoryDetailView(model: browserModel, name: name)
                    }
                }
        }
        .background(Color(.systemGroupedBackground).ignoresSafeArea())
        .onAppear {
            syncBrowserTab(with: page)
            browserModel.updateConnection(chatViewModel.client.canSend)
        }
        .onChange(of: page) { _, newPage in
            // A page switch never carries a push from the page it left —
            // "a page switch has no back button" (the approved mock).
            browserModel.selection = nil
            syncBrowserTab(with: newPage)
            browserModel.contextChanged()
        }
        .onChange(of: chatViewModel.client.canSend) { _, connected in
            browserModel.updateConnection(connected)
        }
        .onChange(of: chatViewModel.client.browserMemoryRevision) { _, _ in
            if page == .memory { browserModel.scheduleRefresh() }
        }
        .onChange(of: chatViewModel.client.browserRunRevision) { _, _ in
            if page == .prompts || browserModel.selection != nil { browserModel.scheduleRefresh() }
        }
        .onDisappear { browserModel.stop() }
        .environment(\.shellOpenPrompts) { runID in
            browserModel.linkedRunID = runID
            page = .prompts
        }
    }

    // MARK: Pages

    /// One identifier per page root (#3106), applied here rather than
    /// inside each page's own view — `AppShellView` is the one place that
    /// already knows which page is showing. `.accessibilityElement(children:
    /// .contain)` first: without it, `.accessibilityIdentifier` broadcasts
    /// onto every leaf element inside a multi-element subtree (confirmed —
    /// it overwrote the composer's own text-field and attach-button ids
    /// with the page root's), not just a single root node.
    private func content(for tab: TopTab) -> some View {
        pageContent(for: tab)
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier(AXID.tabRoot(tab))
    }

    @ViewBuilder
    private func pageContent(for tab: TopTab) -> some View {
        switch tab {
        case .chat:
            MessageView(viewModel: chatViewModel, dockResolveProgress: dockResolveProgress)
        case .memory:
            DataBrowserMemoriesView(model: browserModel, reconnect: chatViewModel.client.reconnect)
        case .prompts:
            DataBrowserPromptsView(model: browserModel, reconnect: chatViewModel.client.reconnect)
        case .settings:
            SettingsView(client: chatViewModel.client)
        }
    }

    private func syncBrowserTab(with tab: TopTab) {
        switch tab {
        case .memory: browserModel.tab = .memories
        case .prompts: browserModel.tab = .prompts
        case .chat, .settings: break
        }
    }

    /// The stack's one push/pop path: a memory selection pushes
    /// `.memoryDetail`, and popping (the system back button, or a page
    /// switch resetting it above) clears the selection — the same binding
    /// shape `DataBrowserMemoriesView` owned itself before the shell became
    /// one shared stack.
    private var routePath: Binding<[ShellRoute]> {
        Binding(
            get: { browserModel.selection.map { [.memoryDetail($0)] } ?? [] },
            set: { newPath in
                guard case .memoryDetail(let name) = newPath.last else {
                    browserModel.selection = nil
                    return
                }
                browserModel.selection = name
            }
        )
    }

    // MARK: Toolbar

    @ToolbarContentBuilder
    private var pageToolbar: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) {
            Menu {
                Picker("Page", selection: $page) {
                    ForEach(TopTab.allCases) { tab in
                        Label(tab.name, systemImage: tab.symbol).tag(tab)
                    }
                }
            } label: {
                Image(systemName: page.symbol)
            }
            .accessibilityIdentifier(AXID.pageMenu)
            .badge(chatViewModel.client.backgroundProgressRuns.count)
        }
        ToolbarItem(placement: .principal) {
            ShellTitle()
        }
        DefaultToolbarItem(kind: .search, placement: .topBarTrailing)
    }
}

/// The toolbar's principal title: Penny's avatar with the small green
/// connected dot, plus her name — a stock composition riding the toolbar's
/// `.principal` slot on every page, not a hand-built chrome row.
private struct ShellTitle: View {
    var body: some View {
        HStack(spacing: 8) {
            ChatTabAvatar()
                .frame(width: 28, height: 28)
                .overlay(alignment: .bottomTrailing) {
                    ZStack {
                        Circle().fill(.white)
                        Circle().fill(.green).frame(width: 8, height: 8)
                    }
                    .frame(width: 11, height: 11)
                    .offset(x: 2, y: 2)
                }
            Text("Penny")
                .font(.headline)
        }
    }
}

/// Penny's avatar for the chat page's principal title — the app owns the
/// `penny` asset.
struct ChatTabAvatar: View {
    var body: some View {
        Image("penny")
            .resizable()
            .scaledToFill()
            .clipShape(Circle())
    }
}

// MARK: Environment

/// Opens the Prompts page on a run — the chat long-press menu's "View
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
