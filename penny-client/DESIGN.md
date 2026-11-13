# Penny client design guidance

Penny should feel at home on iPhone and iPad. Use Apple's Human Interface
Guidelines and built-in controls as the starting point. Keep attention on the
conversation and the user's task.

## Native components first

- Prefer SwiftUI `NavigationStack`, `NavigationSplitView` where a list/detail
  layout benefits from it, system toolbars, `List`, `Form`, `Menu`, `Picker`,
  `.searchable`, sheets, alerts and confirmation dialogs.
- Use the system photo/file pickers and camera presentation where applicable.
  Preserve standard back navigation, dismissal, scrolling and keyboard behavior.
- Reuse `PennyDesign` components and tokens. Create custom UI only for a
  meaningful need that native components do not cover; avoid recreating system
  navigation bars, pickers, buttons or gesture handling.
- Let standard controls supply platform materials and Liquid Glass behavior.
  Avoid decorative glass layers on reading content or effects that reduce
  contrast. Use system typography, semantic colors and spacing consistently.
- “Built-in widgets” means native interface components here; it does not require
  a new WidgetKit extension.

## Symbols and copy

- Prefer familiar SF Symbols for recognizable actions: search, attach, send,
  close, refresh and settings. Match symbol weight and scale to surrounding text,
  and verify availability on the app's minimum OS. See Apple's
  [SF Symbols guidance](https://developer.apple.com/design/human-interface-guidelines/sf-symbols).
- Build controls with semantic labels, such as `Label("Search", systemImage:
  "magnifyingglass")`; use an icon-only visual style when the meaning is clear.
  Every icon-only control must retain a meaningful accessibility label.
- Keep short text labels for ambiguous actions, menu choices and destructive
  confirmations. A symbol must not make someone guess what an action will do.
- Remove redundant headings, explanatory subtitles, welcome prose, repeated
  status messages and instructions for obvious controls. Add text only when it
  helps a person choose, understand a state or recover from an error.
- Empty, loading, offline and error states should be brief and actionable.
  Do not expose implementation details unless they help resolve the problem.

## iPhone and iPad

Adapt to the **available window**, including compact and regular widths,
rotation, keyboard presentation and iPad multitasking. Respect safe areas and
readable content widths. These follow Apple's
[layout guidance](https://developer.apple.com/design/human-interface-guidelines/layout).

- Use container geometry and size classes rather than device-name branches or
  `UIScreen.main.bounds` for layout. A narrow iPad window must remain usable.
- Keep chat readable on wide windows; do not stretch bubbles or forms across
  the full display unnecessarily. Allow useful list/detail layouts where they
  improve browsing, while preserving navigation when the window becomes narrow.
- Keep the composer and primary actions reachable with the keyboard visible.
  Allow sheets and popovers to adapt to the presentation context.
- Preserve selection, draft text and navigation state across resizing and
  rotation. Support touch, hardware keyboard and pointer interaction where relevant.

## Accessibility and visual acceptance

- Support Dynamic Type, including accessibility sizes, without clipping content
  or hiding actions. Allow labels to wrap and layouts to stack when needed.
- Provide comfortable touch targets (normally at least 44 × 44 points), logical
  VoiceOver order, useful labels/values and sufficient contrast. Do not use color
  alone to indicate connection, progress or errors.
- Respect light/dark appearance, Reduce Motion and Reduce Transparency. Use
  native behavior where possible. Apple's
  [accessibility guidance](https://developer.apple.com/design/human-interface-guidelines/accessibility)
  is the baseline.
- Keep automation identifiers in `PennyDesign/AccessibilityIdentifiers.swift`.
  Identifiers locate controls in tests; they do not replace accessible labels.
- For UI changes, inspect local UI-test screenshots on an iPhone and an iPad,
  including a narrow iPad window, large text and relevant keyboard/error states.
  Passing assertions alone do not establish visual quality.

`Design/screens/` contains existing visual references. Use them for context;
these rules and current Apple behavior guide new work when a reference conflicts.
