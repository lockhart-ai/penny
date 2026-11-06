#if DEBUG
import SwiftUI
import PennyDesign
import PennyServices

/// Renders one screen from fixtures, chosen by `-PennyStory <id>` (#2965).
/// `make client-stories` screenshots every `StoryID` case through this view.
struct StoryRootView: View {
    let storyID: StoryID

    var body: some View {
        switch storyID {
        case .chatEmpty:
            AppShellView(chatViewModel: StoryFixtures.chatViewModel(messages: []))
        case .chatConversation:
            AppShellView(chatViewModel: StoryFixtures.chatViewModel(messages: StoryFixtures.conversationMessages))
        case .chatWorking:
            AppShellView(chatViewModel: StoryFixtures.workingViewModel())
        case .chatDisconnected:
            AppShellView(chatViewModel: StoryFixtures.disconnectedViewModel())
        case .chatCollectorsBadge:
            AppShellView(chatViewModel: StoryFixtures.collectorsBadgeViewModel())
        case .shellMemory:
            AppShellView(
                initialTab: .memory,
                chatViewModel: StoryFixtures.chatViewModel(messages: []),
                browserModel: DataBrowserFixtures.model()
            )
        case .shellMemoryDetail:
            AppShellView(
                initialTab: .memory,
                chatViewModel: StoryFixtures.chatViewModel(messages: []),
                browserModel: DataBrowserFixtures.model(selection: "price-watch")
            )
        case .shellMemoryActivity:
            AppShellView(
                initialTab: .memory,
                chatViewModel: StoryFixtures.chatViewModel(messages: []),
                browserModel: DataBrowserFixtures.model(selection: "price-watch", panel: .activity)
            )
        case .shellMemoryConfig:
            AppShellView(
                initialTab: .memory,
                chatViewModel: StoryFixtures.chatViewModel(messages: []),
                browserModel: DataBrowserFixtures.model(selection: "price-watch", panel: .config)
            )
        case .shellPrompts:
            AppShellView(
                initialTab: .prompts,
                chatViewModel: StoryFixtures.chatViewModel(messages: []),
                browserModel: DataBrowserFixtures.model(tab: .prompts)
            )
        case .shellPromptsExpanded:
            AppShellView(
                initialTab: .prompts,
                chatViewModel: StoryFixtures.chatViewModel(messages: []),
                browserModel: DataBrowserFixtures.model(
                    tab: .prompts,
                    expanded: ["run:preview-worked", "prompt:preview-worked:1", "prompt:preview-worked:1:turn:1"]
                )
            )
        case .shellPromptsExpandedChat:
            AppShellView(
                initialTab: .prompts,
                chatViewModel: StoryFixtures.chatViewModel(messages: []),
                browserModel: DataBrowserFixtures.model(
                    tab: .prompts,
                    expanded: ["run:preview-chat", "prompt:preview-chat:1", "prompt:preview-chat:2", "prompt:preview-chat:2:turn:1"]
                )
            )
        case .shellSettings:
            AppShellView(
                initialTab: .settings,
                chatViewModel: StoryFixtures.chatViewModel(messages: StoryFixtures.conversationMessages)
            )
        case .sheetSearch:
            MessageView(viewModel: StoryFixtures.chatViewModel(messages: StoryFixtures.conversationMessages), isShowingSearchStory: true)
        case .sheetActivity:
            MessageView(viewModel: StoryFixtures.workingViewModel(), isShowingActivity: true)
        case .sheetCardDetail:
            MessageView(
                viewModel: StoryFixtures.chatViewModel(messages: StoryFixtures.conversationMessages),
                presentedCardMessage: StoryFixtures.conversationMessages[1]
            )
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
#endif
