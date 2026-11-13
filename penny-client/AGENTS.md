# Penny iOS client: agent guide

This guide applies to `penny-client/`. Start with [README.md](README.md) for the
current project summary, [DESIGN.md](DESIGN.md) for UI decisions, and the
[client workflow](../docs/ios-agent-workflow.md) and
[review guide](../docs/ios-agent-review-guide.md) for implementation and review.

## Project overview

PennyClient is the native SwiftUI iPhone and iPad front end for Penny, a personal
assistant hosted by the backend elsewhere in this repository. It presents shared
chat history, image attachments, live agent progress, permissions, memory and
prompt browsing, and connection settings. The client connects over WebSocket;
it does not run the language model itself.

| Location | Responsibility |
| --- | --- |
| `PennyClient/AppDelegate.swift` | SwiftUI entry point, debug story routing, notification registration |
| `PennyClient/Views/` | App shell, chat, memory/prompts, settings, observable feature models |
| `PennyClient/Stories/` | Debug-only fixtures and named screen states |
| `PennyDesign/` | Shared SwiftUI components, design tokens, accessibility identifiers (`AXID`) |
| `PennyServices/` | WebSocket protocol, synchronization, SQLite storage, search, credentials and preferences |
| `SQLPropertyMacros/` | Local Swift macro package used by persistence |
| `PennyClientTests/` | App model, parsing and presentation unit tests |
| `PennyServices/Tests/` | Service tests, also exposed through the standalone Xcode test target |
| `PennyClientUITests/` | Local-only fixture-driven UI automation |

## Working rules

- Keep changes in the responsible layer. Reuse the existing service injection
  seams and `@MainActor` / `@Observable` feature models; keep networking and
  persistence out of view bodies. Cancel obsolete asynchronous work and prevent
  stale responses from replacing newer state.
- Use native Apple controls and SF Symbols, minimal useful copy, accessible
  labels, and layouts that adapt to both iPhone and iPad. Follow `DESIGN.md`.
  Check the workflow's [UI regression guidance](../docs/ios-agent-workflow.md#ui-regression-guidance)
  when changing controls, navigation, text layout or stories.
- The app targets iOS/iPadOS 26.0, although project-level settings currently say
  27.0. Package minimums differ; check the affected target before choosing APIs.
  Do not raise deployment targets or migrate Swift language modes incidentally.
- Use `PennyClient` to build/run the `PennyDev` app; `PennyTestflight` is the
  distribution scheme. Use `PennyServicesStandaloneTests` for service tests and
  `PennyClientUITests` for local UI tests. Preserve their separation.
- Run UI tests locally to exercise changed flows and retain screenshots. Never
  add UI tests or screenshot generation to CI, `PennyClient.xctestplan`, or the
  main scheme's test action. Keep unit/service tests in CI.
- For Swift or build changes, run SwiftLint with the client and package configs
  in strict mode and require zero violations and zero build/test warnings. Inspect
  full diagnostics; do not suppress warnings to pass. See the workflow's gate.
- Prefer the connected **Sirius** device, rediscovering its identifier each
  session; otherwise use the latest available compatible simulator. Also verify
  the other device family. Details and the observed UUID are in the workflow.
- Use debug stories (`-PennyStory <id>`) for reproducible visual work. Keep story
  hooks out of release builds. Do not use live messages or credentials as fixtures.
- Preserve user edits, local data, signing configuration and secrets. Do not
  erase a personal device or another session's simulator. Read script side
  effects before invoking convenience commands.
- Run checks appropriate to the change, inspect their results, and report what
  actually ran. Documentation-only changes need link/content checks, not an app
  build. Update these docs when architecture, commands or policies change.

## Logging conventions

- Use the existing `LogService` / `OSLogService` abstraction for service logging;
  avoid adding `print`, `NSLog`, direct `os_log` calls or a parallel logging system.
  Keep logging in the owning layer and choose the category and severity deliberately.
- Keep dynamic values private by default. Use public visibility only for known
  non-sensitive operational metadata. `OSLogService` applies privacy to the whole
  message, so every interpolated value must be safe before marking it public.
- Never log message bodies, credentials, authorization headers, pairing/device
  secrets, APNs tokens, attachment contents or raw SQL containing user data.
  Private visibility is not permission to record these values. Check error
  descriptions and URLs for sensitive content before logging them.
- Record concise network and database timings at debug level: measure requests
  from send to the matching response, and SQLite work around the actual operation.
  Use a monotonic clock for elapsed time and safe operation names rather than
  payloads. `OSLogService` already supplies UTC ISO-8601 timestamps.
