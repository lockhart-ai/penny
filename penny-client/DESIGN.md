# PennyClient Design

## The first rule: stock iOS

The client uses SwiftUI's stock elements as Apple designed them, never a hand-built lookalike; our own elements are composed from Apple's components and materials, and every rendering and animation Apple provides (glass, blur, scroll-edge fades, menus, transitions) is Apple's, never faked. Any design rule below that a stock element contradicts loses. See the [client review guide's first section](../docs/penny-client-pr-review-guide.md#1-use-the-framework-as-designed).

Every redesigned screen is built from one place: the [`PennyDesign`](PennyDesign/Package.swift)
Swift package (the tokens and shared components) and the rules below. The approved
reference screens live in [`Design/screens/`](Design/screens/) — every screen in this
document links to its PNG, and a redesigned page is measured against its shot, not
against memory.

## The rules

1. **Three layers.** A grey background (`systemGroupedBackground`); solid white or
   blue content on it (cards, rows, bubbles); glass only for **floating controls** —
   the nav row, the message field, the live dock. Glass never sits on glass, and
   nothing inside a card is glass.
2. **Navigation (#3111).** One stock `NavigationStack` with a visible system bar. A
   leading glass `Menu`/`Picker` switches pages (Chat, Memory, Prompts, Settings) —
   closed, its label is the current page's icon, iOS drawing its own glass circle;
   open, each row carries its icon, name and a checkmark on the current one. Penny's
   avatar and name sit centred (`.principal`) on every page; search is trailing
   (`DefaultToolbarItem(kind: .search)`), each page supplying its own `.searchable`.
   A page switch swaps the content under the same bar, with no back button. A
   drill-down (today, a memory's detail) pushes onto the stack with the system back
   button — nothing hidden, nothing floated over a system element.
3. **Settings save as you change them.** No Save or Done button anywhere.
4. **The rainbow edge is only on the live dock** (the chat dock while Penny works).
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

Screen margin 16; message gap 12; bubble corner 20. No magic numbers in views — a
view that needs a constant adds it to `Tokens` first.

## The components

Memory, Prompts and Settings are redesigned from stock SwiftUI elements directly
(`List`, `Section`, `Label`, `LabeledContent`) under the first rule, rather than a
shared `PennyDesign` view — there is no non-chat kit component yet. Chat components
— bubbles, the message field, the live dock, find-in-chat — come with their own
redesign tickets (#3038–#3045) and land in the kit as they are built.

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
