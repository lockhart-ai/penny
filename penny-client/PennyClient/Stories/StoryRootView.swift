#if DEBUG
import SwiftUI
import PennyServices

/// Renders one screen from fixtures, chosen by `-PennyStory <id>` (#2965).
/// `make client-stories` screenshots every `StoryID` case through this view.
struct StoryRootView: View {
    let storyID: StoryID

    var body: some View {
        switch storyID {
        case .chatEmpty:
            MessageView(viewModel: StoryFixtures.chatViewModel(messages: []))
        case .chatConversation:
            MessageView(viewModel: StoryFixtures.chatViewModel(messages: StoryFixtures.conversationMessages))
        case .chatWorking:
            MessageView(viewModel: StoryFixtures.workingViewModel())
        case .chatDisconnected:
            MessageView(viewModel: StoryFixtures.disconnectedViewModel())
        case .sheetSettings:
            MessageView(viewModel: StoryFixtures.settingsOpenViewModel())
        case .sheetSearch:
            MessageView(viewModel: StoryFixtures.chatViewModel(messages: StoryFixtures.conversationMessages), isShowingSearchStory: true)
        case .sheetActivity:
            MessageView(viewModel: StoryFixtures.workingViewModel(), isShowingActivity: true)
        case .sheetCardDetail:
            MessageView(
                viewModel: StoryFixtures.chatViewModel(messages: StoryFixtures.conversationMessages),
                presentedCardMessage: StoryFixtures.conversationMessages[1]
            )
        case .dataBrowserMemories:
            DataBrowserStoryView(model: DataBrowserFixtures.model())
        case .dataBrowserEntries:
            DataBrowserStoryView(model: DataBrowserFixtures.model(selection: "price-watch"))
        case .dataBrowserActivity:
            DataBrowserStoryView(model: DataBrowserFixtures.model(selection: "price-watch", panel: .activity))
        case .dataBrowserConfig:
            DataBrowserStoryView(model: DataBrowserFixtures.model(selection: "price-watch", panel: .config))
        case .dataBrowserPrompts:
            DataBrowserStoryView(model: DataBrowserFixtures.model(tab: .prompts))
        case .dataBrowserPromptExpanded:
            DataBrowserStoryView(model: DataBrowserFixtures.model(
                tab: .prompts,
                expanded: ["run:preview-worked", "prompt:preview-worked:1", "prompt:preview-worked:1:turn:1"]
            ))
        case .dataBrowserPromptExpandedChat:
            DataBrowserStoryView(model: DataBrowserFixtures.model(
                tab: .prompts,
                expanded: ["run:preview-chat", "prompt:preview-chat:1", "prompt:preview-chat:2", "prompt:preview-chat:2:turn:1"]
            ))
        case .kitTopTabBar:
            KitTopTabBarStory()
        case .kitPageHeader:
            KitPageHeaderStory()
        case .kitListRows:
            KitListRowsStory()
        case .kitPillsTiles:
            KitPillsTilesStory()
        case .kitCodeBlock:
            KitCodeBlockStory()
        }
    }
}

/// Mirrors `DataBrowserPreview` in `DataBrowserPreviews.swift` — the same
/// fixture-backed wrapper around `DataBrowserContent`, reused here instead of
/// duplicated, since that one stays `private` to its preview file.
private struct DataBrowserStoryView: View {
    @State var model: DataBrowserViewModel

    var body: some View {
        DataBrowserContent(model: model, reconnect: {})
            .onAppear { model.updateConnection(true) }
            .onDisappear { model.stop() }
    }
}
#endif
