# Penny for iPhone and iPad

PennyClient is the native app for talking to Penny, your personal AI assistant.
It connects to your Penny server and brings chat, shared conversation history,
images, live activity, memories and prompt inspection into one SwiftUI interface.
The server runs the assistant; the app handles presentation, local history and
communication.

## What is here

- **Chat:** messages, image attachments, search, queued messages, permission
  requests and live agent activity.
- **Memory:** browse memories and their details, configuration and activity.
- **Prompts:** inspect runs and their prompt content.
- **Settings:** server connection, credentials, image options, history and
  notification controls.

The shell uses a system navigation stack and a page menu for Chat, Memory,
Prompts and Settings. Reusable UI lives in `PennyDesign`; WebSocket communication,
SQLite persistence, synchronization, search and preferences live in
`PennyServices`. `SQLPropertyMacros` supports the database layer.

## Open and run

1. Open `PennyClient.xcodeproj` in Xcode and let Swift packages resolve. The local
   packages require Swift tools 6.1/6.2; use an Xcode version that supports them
   and your destination OS. This checkout was inspected with Xcode 27.2.
2. Select **PennyClient** to run the development app, **PennyDev**. The app targets
   iOS/iPadOS 26.0 and enables both iPhone and iPad. Project-level settings say
   27.0; target overrides and package minimums are not uniform.
3. Select a compatible device or simulator. Physical devices need working local
   signing, trust and Developer Mode. Prefer Sirius when available; see the
   [device workflow](../docs/ios-agent-workflow.md#choose-a-destination).
4. For live use, enter your Penny WebSocket URL and credentials in Settings.
   Credentials are stored in Keychain; ordinary preferences use UserDefaults.
   Do not commit credentials or put them in screenshots.

To explore without a server, use a Debug build and add the launch arguments
`-PennyStory chatConversation` in Xcode's scheme editor. Other states are listed
in `PennyClient/Stories/StoryID.swift`. Story mode uses an in-memory message
database and a no-op WebSocket transport, and skips push registration. Remove
the arguments to return to normal use. Some settings still use shared preferences;
story mode is not a complete sandbox for every settings action.

## Checks and screenshots

Run these from the **repository root**:

| Command | Purpose |
| --- | --- |
| `make client-check` | CI-equivalent service/app unit tests and TestFlight build metadata check |
| `make client-services-check` | Standalone service tests |
| `make client-ui-tests` | Local-only UI interaction tests on a simulator |
| `make client-stories` | Existing local story gallery in `penny-client/build/stories/` |

The check scripts erase their selected simulator; `client-check` and
`client-services-check` also shut down all simulators. Coordinate before using
them on a shared machine. For an explicit destination, device testing and retained
UI-test screenshots, follow the [workflow](../docs/ios-agent-workflow.md).
UI tests and screenshot capture stay out of CI.

## Assessment: November 13, 2026

The source has useful separation between app, reusable UI and services, with
unit coverage and fixture-driven UI tests for navigation, composer, search and
connection settings. CI already keeps UI automation separate from unit tests.

The remaining workflow and design gaps are:

- UI tests currently have no explicit named screenshot attachments. The workflow
  describes how to add and retain them for local visual review.
- `client-ui-tests.sh` selects the newest available iPhone simulator, but does
  not prefer a physical device or cover iPad automatically. `client-stories.sh`
  pins iPhone 18 Pro / iOS 27.2 and uses a fixed delay before capture.
- Universal device support is configured, but is not evidence of layout quality
  at every size. For example, `PennyDesign/LiveDock.swift` bases its height cap on
  `UIScreen.main.bounds`, which needs review for resized iPad windows.
- Project/target deployment settings differ, and GitHub workflows select Xcode
  26 while this local installation is Xcode 27.2. Check compatibility when changing
  build settings or adopting APIs.

This is a source, configuration and tool-inventory assessment, not a fresh build
or runtime certification. No app build, test suite or device installation was
performed for this documentation change.

## Working on the app

- [Agent project guide](AGENTS.md)
- [Design guidance](DESIGN.md)
- [Agent workflow and screenshots](../docs/ios-agent-workflow.md)
- [Concise review guide](../docs/ios-agent-review-guide.md)
