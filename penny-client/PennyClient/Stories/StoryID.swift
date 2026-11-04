#if DEBUG
/// Every screen `make client-stories` renders from fixtures and screenshots
/// (#2965). `scripts/client-stories.sh` reads this file's `case` lines as the
/// one list of story ids — never a second, hand-kept list in the script.
enum StoryID: String, CaseIterable {
    case chatEmpty
    case chatConversation
    case chatWorking
    case chatDisconnected
    case sheetSettings
    case sheetSearch
    case sheetActivity
    case sheetCardDetail
    case dataBrowserMemories
    case dataBrowserEntries
    case dataBrowserActivity
    case dataBrowserConfig
    case dataBrowserPrompts
    case dataBrowserPromptExpanded
    case dataBrowserPromptExpandedChat
}
#endif
