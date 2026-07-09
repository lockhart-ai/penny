# PennyClient Instructions

## Project Context
PennyClient is a SwiftUI iOS chat client that connects to the Penny websocket service. The main app entry point and chat UI live in `PennyClient/PennyClient/Views/MessageView.swift`, with screen state and actions in `PennyClient/PennyClient/Views/MessageView+ViewModel.swift`.
Use Xcode project-relative paths when working from Xcode tooling. Prefer Xcode tools for reading, editing, diagnostics, and builds.

## Important Files
- `PennyClient/PennyClient/Views/MessageView.swift`: Main chat screen, toolbar, composer, message rows.
- `PennyClient/PennyClient/Views/MessageView+ViewModel.swift`: Message screen view model.
- `PennyClient/PennyClient/Views/SettingsView.swift`: Editable connection settings.
- `PennyClient/PennyClient/Service/PennyWebSocketClient.swift`: Websocket connection, registration, message handling, badge clearing.
- `PennyClient/PennyClient/Service/DatabaseService.swift`: SQLite-backed message persistence.
- `PennyClient/PennyClient/Service/Prefs.swift`: UserDefaults wrapper and optional `Secrets.plist` loading.
- `PennyClient/PennyClient/AppDelegate.swift`: Push notification registration and foreground notification handling.
- `PennyClient/PennyClient/PennyDev.entitlements`: PennyDev push notification entitlements.
- `PennyClient/PennyClient/PennyTestflight.entitlements`: PennyTestflight push notification entitlements.

## Secrets And Preferences
Connection settings are managed by `Prefs`.
Lookup order:
1. Saved `UserDefaults` values.
2. Optional bundled `Secrets.plist`.
3. `nil` when neither exists.

The real `PennyClient/PennyClient/Secrets.plist` is local-only and ignored by git. Keep `PennyClient/PennyClient/Secrets.plist.example` updated with the expected keys:
- `webSocketURL`
- `username`
- `password`

## Build And Validation
Use `BuildProject` for full validation. Use `XcodeRefreshCodeIssuesInFile` for fast Swift diagnostics, but treat stale cross-file scope errors cautiously if the full build and Issue Navigator are clean.
Zero warnings are expected. Treat all warnings as issues to fix before finishing, including SwiftLint warnings; do not leave new or existing SwiftLint warnings unresolved.
For final verification, prefer:
- diagnostics on touched Swift files
- `XcodeListNavigatorIssues` for workspace errors
- focused tests for touched behavior, or `RunAllTests` when the change affects shared behavior
- `BuildProject`

## Testing Guidelines
Use the Swift Testing framework for unit and component tests. UI tests are not recommended by default; avoid XCUIAutomation unless the behavior cannot be validated with focused unit, component, or service-level tests.
Keep tests organized by component in separate files under `PennyClient/PennyClientTests`, with shared helpers in test support files instead of large monolithic test sources.

All view models require test coverage. When adding or changing a view model, add or update focused tests for:
- state derived from model/client data
- user actions and side effects
- validation and empty-input behavior
- filtering, sorting, or presentation-specific computed values
- async lifecycle behavior when relevant

Prefer injecting test doubles or configured in-memory services over relying on live network, real user defaults, or persistent app data. For persistence tests, use `DatabaseService.setupForTesting()` or another isolated in-memory setup.

## UI Patterns
Use SwiftUI and the Observation framework. Avoid Combine unless there is a concrete need.
The chat UI uses:
- `NavigationStack`
- a custom toolbar title with the Penny image, title, and status dot
- a settings sheet from the gear icon
- a bottom overlay composer with keyboard-aware offset
- split Markdown text blocks for incoming messages
- attachment images inside incoming message bubbles
Keep toolbar icons borderless/plain unless a visible Liquid Glass control is intentionally requested.

## Websocket Notes
`PennyWebSocketClient` reads connection configuration from `Prefs.shared` at connect time. Saving Settings should reconnect the client so URL/auth changes take effect.
The websocket request uses HTTP Basic auth from the saved username/password. The websocket URL should usually be `wss://...`, not `https://...`.
The client registers with a stable keychain-backed device id and device secret, sends APNs token updates, pulls pending messages, acknowledges received message ids, and clears badges after messages are received.

## Persistence Notes
Messages are stored through `DatabaseService` using SQLite.swift and SQLPropertyMacros. Attachments are persisted as JSON-encoded data URL strings on the message model.
