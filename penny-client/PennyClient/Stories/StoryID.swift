#if DEBUG
/// Every screen `make client-stories` renders from fixtures and screenshots
/// (#2965). `scripts/client-stories.sh` reads this file's `case` lines as the
/// one list of story ids — never a second, hand-kept list in the script.
enum StoryID: String, CaseIterable {
    case chatEmpty
    case chatConversation
    case chatBubbles
    case chatWorking
    case chatDisconnected
    case shellMemory
    case shellMemoryDetail
    case shellMemoryActivity
    case shellMemoryConfig
    case shellPrompts
    case shellPromptsExpanded
    case shellPromptsExpandedChat
    case shellSettings
    case sheetSearch
    case sheetActivity
    case sheetCardDetail
    case kitTopTabBar
    case kitPageHeader
    case kitListRows
    case kitPillsTiles
    case kitCodeBlock
}
#endif
