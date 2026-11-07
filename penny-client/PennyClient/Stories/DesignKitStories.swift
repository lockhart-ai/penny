#if DEBUG
import SwiftUI
import PennyDesign

/// The design kit's stories (#3036): one screen per component group, all
/// from contrived content, so `make client-stories` renders every kit
/// component on a plain page. Nothing here touches live services.
@MainActor
enum DesignKitStories {
    /// The page background every non-chat page sits on.
    static var pageBackground: some View {
        Color(.systemGroupedBackground).ignoresSafeArea()
    }
}


/// The page header and the section shapes: `PageTitle`, the grey
/// `SectionHeader` above every white `SectionCard`, hairline dividers, and
/// a footer.
struct KitPageHeaderStory: View {
    @State private var autosave = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Tokens.messageGap) {
                PageTitle(
                    title: "Memory",
                    byline: "42 collections · 1,208 entries",
                    symbol: "books.vertical",
                    tint: .blue
                ) {
                    GlassCircleButton(symbol: "magnifyingglass")
                }
                SectionCard(header: "Collections", rows: [
                    ListRow(
                        icon: .kind(MemoryKind.collection),
                        title: "Price watch",
                        subtitle: .standard("Keeps an eye on the coffee grinder's price at three shops."),
                        caption: .text("Every 6 hours", trailing: "3,201 entries"),
                        showsChevron: true
                    ),
                    ListRow(
                        icon: .kind(MemoryKind.collection),
                        title: "Reading list",
                        caption: .text("Every morning", trailing: "412 entries"),
                        showsChevron: true
                    ),
                    ListRow(
                        icon: .kind(MemoryKind.log),
                        title: "Activity log",
                        trailing: .time("12:04")
                    ),
                    ListRow(
                        icon: .kind(MemoryKind.archived),
                        title: "Archived",
                        trailing: .value("7")
                    )
                ])
                SectionCard(header: "Details", footer: "Saving happens as fields change.") {
                    VStack(alignment: .leading, spacing: Tokens.rowLineGap) {
                        ListRow(
                            title: "Notifications",
                            trailing: .toggle($autosave)
                        )
                        Text("Collections run on their schedule and write entries; Penny reads them when asked. Nothing here needs a Save button.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.horizontal, Tokens.margin)
            .padding(.vertical, Tokens.margin)
        }
        .background(DesignKitStories.pageBackground)
    }
}

/// The one list row in its variants: kinds, statuses, trailing items,
/// badges, metrics and a live row's progress.
struct KitListRowsStory: View {
    @State private var notify = true
    @State private var mute = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Tokens.messageGap) {
                PageTitle(
                    title: "Rows",
                    byline: "Every variant of the one list row",
                    symbol: "list.bullet.rectangle",
                    tint: .teal
                )
                SectionCard(header: "Kinds", rows: RunKind.allCases.map { kind in
                    ListRow(
                        icon: .kind(kind),
                        title: kind.name,
                        trailing: .time("11:38"),
                        subtitle: .standard("A run's kind sets its tile colour and icon.")
                    )
                })
                SectionCard(header: "Statuses", rows: [
                    ListRow(
                        icon: .kind(RunKind.collector),
                        title: "Inbox sweep",
                        pill: StatusPill(status: .running),
                        trailing: .running,
                        caption: .text("Reading 12 of 40 messages"),
                        progress: 0.3
                    ),
                    ListRow(
                        icon: .kind(RunKind.collector),
                        title: "Price watch",
                        pill: StatusPill(status: .worked),
                        trailing: .time("09:15")
                    ),
                    ListRow(
                        icon: .kind(RunKind.history),
                        title: "Hourly news",
                        pill: StatusPill(status: .nothingNew),
                        trailing: .time("08:00")
                    ),
                    ListRow(
                        icon: .kind(RunKind.collector),
                        title: "Archive sweep",
                        pill: StatusPill(status: .failed),
                        trailing: .time("Yesterday")
                    ),
                    ListRow(
                        icon: .kind(RunKind.chat),
                        title: "Flagged run",
                        pill: StatusPill(status: .flagged),
                        trailing: .time("Tue")
                    )
                ])
                SectionCard(header: "Trailing items", rows: [
                    ListRow(
                        icon: ListRowTile.tile(symbol: "bell.badge.fill", tint: .red),
                        title: "Notifications",
                        trailing: .toggle($notify)
                    ),
                    ListRow(
                        icon: ListRowTile.tile(symbol: "bell.slash.fill", tint: .gray),
                        title: "Muted",
                        trailing: .toggle($mute)
                    ),
                    ListRow(
                        icon: ListRowTile.tile(symbol: "doc.text.fill", tint: .teal, badge: 3),
                        title: "Entry",
                        subtitle: .monospaced("price_watch.2026-11-05T09:15:04Z"),
                        showsChevron: true
                    ),
                    ListRow(
                        icon: .kind(TurnRole.toolCall),
                        title: "Tool call",
                        subtitle: .standard("browse · 2 arguments · 1.4 s"),
                        caption: .text("Turn 2 of 5", trailing: "steps 3–4"),
                        showsChevron: true
                    )
                ])
                SectionCard(header: "Metrics", rows: [
                    ListRow(
                        icon: .kind(RunKind.collector),
                        title: "Last cycle",
                        caption: .metrics(["1,204", "890", "314", "2.1 tok/s", "48 s"])
                    )
                ])
            }
            .padding(.horizontal, Tokens.margin)
            .padding(.vertical, Tokens.margin)
        }
        .background(DesignKitStories.pageBackground)
    }
}

/// The small parts: `IconTile` at its sizes and colours, every
/// `StatusPill` status, and the `MetricsStrip`.
struct KitPillsTilesStory: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Tokens.messageGap) {
                PageTitle(
                    title: "Tiles & pills",
                    byline: "Icon tiles, status pills, metric columns",
                    symbol: "square.grid.2x2",
                    tint: .purple
                )
                SectionCard(header: "Icon tiles", content: {
                    VStack(alignment: .leading, spacing: Tokens.rowLineGap) {
                        HStack(spacing: Tokens.iconTextGap) {
                            ForEach(RunKind.allCases, id: \.self) { kind in
                                IconTile(symbol: kind.symbol, tint: kind.color)
                            }
                            ForEach(MemoryKind.allCases, id: \.self) { kind in
                                IconTile(symbol: kind.symbol, tint: kind.color)
                            }
                            IconTile(symbol: "globe", tint: .mint, side: Tokens.pageIconSize)
                        }
                        HStack(alignment: .firstTextBaseline, spacing: Tokens.iconTextGap) {
                            ForEach(TurnRole.allCases, id: \.self) { role in
                                IconTile(symbol: role.symbol, tint: role.color)
                            }
                        }
                    }
                })
                SectionCard(header: "Status pills", content: {
                    VStack(alignment: .leading, spacing: Tokens.rowLineGap) {
                        HStack(spacing: 8) {
                            ForEach([RunStatus.worked, .nothingNew, .failed], id: \.self) { status in
                                StatusPill(status: status)
                            }
                        }
                        HStack(spacing: 8) {
                            ForEach([RunStatus.flagged, .running], id: \.self) { status in
                                StatusPill(status: status)
                            }
                            StatusPill(label: "LIVE", color: .green)
                        }
                    }
                })
                SectionCard(header: "Metrics", content: {
                    MetricsStrip(values: ["1,204", "890", "314", "2.1 tok/s", "48 s"])
                })
            }
            .padding(.horizontal, Tokens.margin)
            .padding(.vertical, Tokens.margin)
        }
        .background(DesignKitStories.pageBackground)
    }
}

/// `CodeBlock` in both modes inside the shared white card.
struct KitCodeBlockStory: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Tokens.messageGap) {
                PageTitle(
                    title: "Code blocks",
                    byline: "Raw machine text, read-only",
                    symbol: "curlybraces",
                    tint: .indigo
                )
                SectionCard(header: "Arguments (JSON)") {
                    CodeBlock(
                        text: """
                        {
                          "query": "moka pot grind size",
                          "limit": 5,
                          "sources": ["web", "memory"],
                          "fresh": true,
                          "older_than": null,
                          "weight": 0.75
                        }
                        """,
                        mode: .json
                    )
                }
                SectionCard(header: "Thinking (prose)") {
                    CodeBlock(
                        text: """
                        ## Plan
                        The user wants a grind size for a moka pot.
                        - Look up their saved brewing notes first
                        - Then browse one roasting guide
                        - Reply with the coarsest workable range
                        """,
                        mode: .markdown
                    )
                }
            }
            .padding(.horizontal, Tokens.margin)
            .padding(.vertical, Tokens.margin)
        }
        .background(DesignKitStories.pageBackground)
    }
}
#endif
