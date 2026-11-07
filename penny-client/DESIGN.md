# PennyClient Design

## The first rule: stock iOS

The client uses SwiftUI's stock elements as Apple designed them, never a hand-built lookalike; a custom view only where no stock element exists. Any design rule below that a stock element contradicts loses. See the [client review guide's first section](../docs/penny-client-pr-review-guide.md#1-use-the-framework-as-designed).

Every redesigned screen is built from one place: the [`PennyDesign`](PennyDesign/Package.swift)
Swift package (the tokens and shared components) and the rules below. The approved
reference screens live in [`Design/screens/`](Design/screens/) — every screen in this
document links to its PNG, and a redesigned page is measured against its shot, not
against memory.

The kit's components each render in story mode (`make client-stories`, story ids
`kit*` in `PennyClient/Stories/StoryID.swift`) — run it and look at the PNGs under
`penny-client/build/stories/` before eyeballing anything by hand.

## The rules

1. **Three layers.** A grey background (`systemGroupedBackground`); solid white or
   blue content on it (cards, rows, bubbles); glass only for **floating controls** —
   the nav row, the message field, the live dock. Glass never sits on glass, and
   nothing inside a card is glass.
2. **Navigation.** One centred glass tab capsule at the top (Chat = Penny's avatar,
   Memory, Prompts, Settings) plus a search circle that searches the current tab.
   Pushed pages keep the capsule and add a glass back button leading.
3. **Every page has an `[icon] Title / byline` header** (`PageTitle`): the page's
   type icon tile, a bold one-line title, and a **required one-line byline** under
   it. There is one title size in the whole app (`Tokens.pageTitleStyle/Weight`);
   content inside cards is never larger than the page title.
4. **A section header above every card** (`SectionCard`): the grey small-caps-style
   `SectionHeader` sits on the header icon's column (`Tokens.sectionHeaderX`), one
   white card under it, hairline dividers between rows. `SectionCard` is the only
   way to make a section — screens never draw their own cards or dividers.
5. **Rows, not tiles.** Every data row is the one `ListRow`: leading type icon tile,
   title, status pill, trailing time/value/toggle, subtitle, caption or metrics
   strip, chevron, and a live row's progress edge. Screens pass data; the row picks
   every font, spacing and alignment.
6. **Icon per type, top-aligned with the title's capitals.** Icons come from the
   fixed per-type tables below — never set per item. Every tile's TOP edge is level
   with the TOP OF THE CAPITAL LETTERS of its title's first line (cap height, from
   real font metrics, computed inside the kit — screens never hand-align).
7. **Settings save as you change them.** No Save or Done button anywhere.
8. **The rainbow edge is only on the live dock** (the chat dock while Penny works).
   It means "Penny is active" and appears nowhere else — `pennyRainbow` is the only
   colour accent outside the semantic palette.

## The type tables (one icon per type, with its colour)

| Run kind (`RunKind`) | Colour |
|---|---|
| Chat | purple |
| Collector | blue |
| History | orange |

| Memory kind (`MemoryKind`) | Colour |
|---|---|
| Collection | blue |
| Log / Archived | gray |
| Entry | teal |

| Turn role (`TurnRole`) | Colour |
|---|---|
| SYSTEM | gray |
| USER | blue |
| THINKING | purple |
| TOOL CALL / RESPONSE | green |
| RESULT | teal |

| Status (`RunStatus`) | Colour |
|---|---|
| WORKED | green |
| NOTHING NEW | gray |
| FAILED | red |
| FLAGGED | orange |
| RUNNING | blue (live) |

## The spacing grid (`Tokens`)

Screen margin 16; card inner padding 12; row vertical padding 11, line gap 3; the
rows' text column `rowTextX` = 16 + 12 + 29 + 12 — the page header's title starts
exactly there, on the same column as every card row's text; icon tile 29 (page
header's tile is the derived `pageIconSize`, spanning the title's cap top down to
the byline's baseline); section header x = the header icon's column; message gap 12;
every glass control in the nav row is 44pt tall (Apple's minimum tap target) with a
19.5pt glyph; card corner 16, bubble corner 20. No magic numbers in views — a view
that needs a constant adds it to `Tokens` first.

## The components

| Component | Kit file | Reads |
|---|---|---|
| `TopTabBar` (tabs, badges, avatar), `GlassCircle`, `GlassCircleButton` | `TopTabBar.swift` | nav control height, glyph size |
| `PageTitle` | `PageTitle.swift` | `pageIconSize`, `pageIconX`, `rowTextX` |
| `SectionHeader`, `SectionCard` | `SectionCard.swift` | `sectionHeaderInset`, `cardInnerPadding`, `cardCorner` |
| `ListRow` (with `ListRowTile`, `RowIconKind`) | `ListRow.swift` | `cardInnerPadding`, `tile`, `iconTextGap`, `rowVerticalPadding`, `rowLineGap` |
| `StatusPill` | `StatusPill.swift` | — |
| `IconTile` | `IconTile.swift` | `tile` |
| `MetricsStrip` | `MetricsStrip.swift` | `metricColumns`, `metricSpacing` |
| `CodeBlock` | `CodeBlock.swift` | `iconTextGap` |

Chat components — bubbles, the message field, the live dock, find-in-chat — come
with their own redesign tickets (#3038–#3045) and land in the kit as they are built.

## Reference screens

Grouped as they appear in `Design/screens/`; the name is the state it shows.

- **Chat:** `chat.png`, `chat-working.png`, `chat-working-start.png`,
  `chat-working-long.png`, `chat-drawing.png`, `chat-drawing-done.png`,
  `chat-pending.png`, `chat-queued.png`, `chat-queued-menu.png`,
  `chat-permission.png`, `chat-attach.png`, `chat-menu.png`,
  `chat-collectors.png`, `chat-resolving.png`, `chat-search.png`,
  `chat-search-empty.png`
- **Runs & turns:** `run.png`, `run-live.png`, `turn.png`
- **Reading raw text:** `read-system.png`, `read-thinking.png`, `read-tool.png`
- **Memory:** `memory.png`, `memory-collection.png`, `memory-entry.png`,
  `memory-settings.png`
- **Prompts:** `prompts.png`
- **Settings:** `settings.png`, `settings-server.png`
