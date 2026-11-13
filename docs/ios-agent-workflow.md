# iOS agent workflow

Scope: development agents working on `penny-client/`. Read the
[project guide](../penny-client/AGENTS.md), [summary](../penny-client/README.md) and
[design rules](../penny-client/DESIGN.md) first. Commands below run from the
repository root. Apple references were checked on November 13, 2026; rediscover
local tools, destinations and API availability as they change.

## Inspect, change, verify

1. Check `git status`, the affected feature, its service boundary and existing
   tests. Identify the user-visible outcome and the smallest useful verification.
   Preserve unrelated edits and follow the active session's branch/worktree policy.
2. Reproduce UI issues with an existing Debug story, or add a focused fixture in
   `PennyClient/Stories/`. Use injected state and deterministic data instead of
   connecting a test to the live assistant.
3. Make a focused change. Reuse native controls, `PennyDesign`, observable models
   and service seams. Preserve actor isolation, cancellation and stable identity;
   avoid network/database work in SwiftUI view bodies. Follow the
   [logging conventions](../penny-client/AGENTS.md#logging-conventions) and the
   UI regression guidance below.
4. Run SwiftLint, build the affected target and run relevant unit/service tests.
   Require zero lint violations and zero build/test warnings, as described below.
   Use local UI tests for the changed interaction and capture the screen states.
5. Inspect screenshots and the test result bundle. Exercise the other device
   family and relevant accessibility/window states. Fix observed failures before
   broadening checks; do not repeatedly rerun unchanged failures without diagnosis.
6. Review with the [short review guide](ios-agent-review-guide.md). Report the
   change, exact checks, device/OS, artifact paths and any unverified behavior.

For new unit tests, consider Swift Testing where it fits the existing target;
keep XCTest for UI automation and performance tests. Do not mix the two assertion
APIs in one test or migrate an existing suite incidentally. This follows Apple's
[testing framework guidance](https://developer.apple.com/documentation/xctest).

## UI regression guidance

These rules preserve lessons from earlier client bugs. Apply them to the affected
flow and add a focused regression check when fixing a recurrence.

- **Wire actions explicitly.** Required component actions such as send or select
  must be required initializer arguments, not silent no-op defaults. Optional
  actions should be modeled explicitly, with an appropriate hidden/disabled state.
  A compiling control is not evidence that its action is connected.
- **Preserve system navigation.** Never hide a system navigation bar and place
  lookalike controls over it; this previously caused dead taps. Keep the page menu
  on root content and preserve the system back button on pushed destinations.
  Do not assume a root toolbar is inherited by every destination.
- **Keep glass identities unique.** Never have two simultaneously presented views
  claim one `glassEffectID`; the live-dock/reply transition previously crashed
  this way. Use the existing stock transitions and test overlapping arrival/removal
  states instead of reintroducing a shared identity.
- **Keep styled paragraphs together.** Do not lay out attributed-string runs as
  separate `Text` views in a flow layout. Preserve paragraph-level wrapping and
  check mixed bold text, links and search highlights at narrow widths and large text.
- **Keep inset backgrounds continuous.** A composer or `safeAreaInset` must not
  reveal a differently colored window behind it. Inspect the boundary while the
  keyboard opens/closes and in both light and dark appearance.
- **Apply story state after loading.** State that depends on asynchronously loaded
  data must be applied after that load completes, not raced against it in
  `onAppear`. Wait for the intended state before asserting or taking a screenshot.
- **Test search end to end.** Open search, type, verify filtering or chat highlights,
  then close it and verify dismissal. Preserve the existing combination of
  `.searchable`, `DefaultToolbarItem(kind: .search)` and
  `.searchToolbarBehavior(.minimize)` unless a tested replacement is intentional;
  the source records a toolbar regression when that combination was incomplete.
- **Exercise added or moved controls.** Update local UI tests to drive the actual
  tap, navigation or send and assert its resulting state. Screenshots supplement
  those assertions; an existence-only check cannot detect a disconnected action.

## SwiftLint and zero-warning gate

For Swift source, dependency or build-configuration changes, completion requires
**zero SwiftLint violations, zero build/test warnings and no errors**. A successful
exit code alone is not sufficient, and “no new warnings” does not meet this gate.

Run SwiftLint with the repository's existing configurations. The client config
does not include `PennyServices`, so lint that package separately; also check
`PennyDesign` with its package configuration. From the repository root:

```sh
set -e
swiftlint version
(cd penny-client && swiftlint lint --strict)
(cd penny-client/PennyDesign && swiftlint lint --strict)
(cd penny-client/PennyServices && swiftlint lint --strict)
```

Use a SwiftLint version consistent with the resolved project dependency and the
same Swift toolchain as the build. `--strict` makes lint warnings fail the command;
see the [SwiftLint documentation](https://github.com/realm/SwiftLint#configuration).
If the CLI is missing, locate the resolved SwiftLint executable or install the
matching version through the normal development setup. Report a blocked lint
check rather than silently skipping it. Build-plugin execution alone does not
prove all three configurations were checked.

For final build/test verification, use a fresh, task-owned `-derivedDataPath` so
an incremental no-op build cannot hide compiler warnings. Add
`SWIFT_TREAT_WARNINGS_AS_ERRORS=YES` and `GCC_TREAT_WARNINGS_AS_ERRORS=YES` to the
`xcodebuild` invocation. Inspect the full build/test logs and result bundle for
remaining Xcode, linker, script, plugin and test warnings; compiler flags do not
cover every diagnostic. If piping output to `tee`, enable `set -o pipefail` so
logging does not mask a failing command.

Fix warnings and rerun the affected checks. Do not weaken lint rules, add a
baseline, suppress diagnostics or change warning severity just to pass. If an
existing or dependency warning cannot be resolved within scope, report its
source and the unmet gate explicitly; do not claim zero warnings. The handoff
must state the SwiftLint result and build/test warning count, with log/result
paths. Documentation-only edits still need documentation checks, not a new build.

## Choose a destination

**Prefer the connected device requested as “Sirus”.** Discovery on November 13,
2026 found it named **Sirius**, connected and available, running **iPadOS 27.0**:

```text
Device UUID reported by XcodeBuildMCP:
5771124D-00F3-5E80-8696-403B3FD55716
```

Recheck the device each session; this is an observed identifier, not a permanent
guarantee of availability. XcodeBuildMCP `list_devices` is the preferred discovery
tool. Its device identifier can be used with its device tools. With the CLI, use:

```sh
xcodebuild -version
xcrun devicectl list devices
xcodebuild -project penny-client/PennyClient.xcodeproj \
  -scheme PennyClientUITests -showdestinations
```

For `xcodebuild -destination`, use the matching **iOS destination ID** from
`-showdestinations`; a CoreDevice UUID and Xcode's hardware UDID can differ. Confirm
the device name and OS rather than assuming any UUID-shaped value is interchangeable.
During this assessment, XcodeBuildMCP discovery succeeded, but the separate CLI
`devicectl` query timed out initializing CoreDeviceService; no device test was run.

Sirius must be paired, trusted, unlocked, have Developer Mode enabled and have
valid development signing. If it is unavailable or cannot run tests, record the
reason and use the **newest installed, available iOS simulator runtime compatible
with the selected Xcode and scheme**. Compare numeric runtime versions, not list
order; do not choose an older runtime merely because its simulator is booted.
For equal versions, prefer the general-purpose runtime over a device-specific
preview runtime unless that device is the subject of the task.

Discover simulators with XcodeBuildMCP `list_sims`, or:

```sh
xcrun simctl list runtimes -j
xcrun simctl list devices available -j
```

At inspection, the newest general-purpose available runtime was iOS 27.2, with
both iPhone and iPad destinations. Treat this as a snapshot, not a version pin.
Choose an iPhone for phone coverage and an iPad for tablet coverage. If Sirius is
used, add iPhone simulator coverage. If it is absent, use both simulator families.
If no compatible runtime is installed, report the missing prerequisite; do not
silently test on an unsupported OS.

Before the first XcodeBuildMCP build/run/test, call `session_show_defaults` and
set the correct project, scheme and device/simulator ID with `session_set_defaults`.
Recheck defaults when switching destinations. Prefer `build_run_device` or
`build_run_sim` for launch, and `test_device` or `test_sim` for testing. These tool
names are the observed interface; inspect available tools before relying on them.

## Local UI tests and screenshots

**UI tests, UI smoke tests, accessibility UI audits and screenshot generation are
local-only. Do not add them to any CI test suite.** Use the dedicated
`PennyClientUITests` scheme. Keep it out of `PennyClient.xctestplan`, the main
scheme's Test action and GitHub/Xcode Cloud test jobs. CI continues to run
unit/service tests and build checks through `make client-check`.

Existing UI tests use `XCUIApplication.launchStory(_:)`, typed element queries and
`AXID` identifiers. Extend those patterns:

- Launch an explicit fixture such as `uiTestChat`, with known data and expected
  outcomes. Story mode avoids live sockets, message storage and push registration,
  but settings still use shared preferences/Keychain in places. Avoid changing
  personal credentials on Sirius; use isolated dependencies or a dedicated
  simulator for settings tests that persist changes.
- Wait for observable state with `waitForExistence`, predicate expectations and
  the helpers in `UITestSupport.swift`. Check that a control is hittable before
  interacting. Do not use fixed sleeps to declare a screen ready.
- Prefer identifiers and element types over coordinates or localized visible
  copy. During agent-driven inspection, observe the current semantic UI before
  acting and refresh that observation after navigation or layout changes.
- Assert the expected state, then attach a named screenshot. Stop at unexpected
  alerts or navigation and inspect logs/hierarchy; avoid blind repeated taps.
- Keep fixtures, appearance, orientation and text size reproducible. Capture
  meaningful before/after states for a UI fix rather than every animation frame.

The existing UI tests do **not** yet add named screenshot attachments. Add this
pattern to the relevant local test after its expected state is reached (example
inside a test method, with `XCTest` and `PennyDesign` imported):

```swift
let app = XCUIApplication()
app.launchStory("uiTestChat")
XCTAssertTrue(app.button(AXID.pageMenu).waitForExistence(timeout: UITestWait.standard))

let capture = XCTAttachment(screenshot: app.screenshot())
capture.name = "chat-ready"
capture.lifetime = .keepAlways
add(capture)
```

Use `.keepAlways` so successful tests retain their images. Group multi-step flows
with `XCTContext.runActivity` when useful. See Apple's
[attachment documentation](https://developer.apple.com/documentation/xctest/adding-attachments-to-tests-activities-and-issues).

For an explicit local test run, first set a discovered destination. Replace the
placeholder with the verified ID; choose one of these assignments:

```sh
DESTINATION='platform=iOS,id=<Sirius destination ID>'
# Simulator alternative:
# DESTINATION='platform=iOS Simulator,id=<selected simulator UUID>'
```

Then run the relevant existing suite, retaining results in an ignored directory.
The validation-skip flags below match the repository scripts for its existing
macro/plugin dependencies; review dependency changes before trusting new code.

```sh
./scripts/wait-for-penny-idle.sh
mkdir -p penny-client/build/review
REVIEW_RUN_DIR=$(mktemp -d "$PWD/penny-client/build/review/run.XXXXXX")
xcodebuild test \
  -project penny-client/PennyClient.xcodeproj \
  -scheme PennyClientUITests \
  -configuration Debug \
  -destination "$DESTINATION" \
  -parallel-testing-enabled NO \
  -only-testing:PennyClientUITests/ShellNavigationUITests \
  -resultBundlePath "$REVIEW_RUN_DIR/UITests.xcresult" \
  -skipMacroValidation \
  -skipPackagePluginValidation
```

Change `-only-testing` to the relevant class/method, or omit it for the whole
local UI suite. After completion, export and open the screenshots, or inspect
attachments directly in Xcode:

```sh
xcrun xcresulttool export attachments \
  --path "$REVIEW_RUN_DIR/UITests.xcresult" \
  --output-path "$REVIEW_RUN_DIR/screenshots"
open "$REVIEW_RUN_DIR/UITests.xcresult"
```

Export syntax was checked against the installed Xcode 27.2 tool. Review the
exported manifest to associate images with tests; opening an artifact and
inspecting its contents is part of verification. A screenshot alone does not
prove an interaction succeeded. Keep generated images, logs and `.xcresult`
bundles out of commits unless explicitly requested.

## Coverage and shared-machine discipline

- For UI changes, check compact iPhone and regular iPad layouts, a narrow iPad
  window, rotation and any relevant keyboard-visible state. Include large
  Dynamic Type and light/dark appearance when layout or styling changes.
- Use VoiceOver/manual inspection for interaction semantics. Local UI tests can
  call `try app.performAccessibilityAudit()` for automated checks; inspect and fix
  reported issues rather than broadly filtering failures. See Apple's
  [audit API](https://developer.apple.com/documentation/xcuiautomation/xcuiapplication/performaccessibilityaudit(for:_:)).
- Use a real device for camera, notification delivery or hardware behavior when
  that is the subject of the change. Mark simulator-only validation explicitly.
- Run `scripts/wait-for-penny-idle.sh` before local builds/tests that compete with
  the model server. Coordinate simulator/device ownership across sessions and
  serialize UI runs on the same destination. Do not erase Sirius or user data.
- Prefer explicit destinations for shared-machine work. Existing
  `client-check.sh` and `client-services-check.sh` shut down **all** simulators
  and erase the selected one; `client-ui-tests.sh` erases its selected simulator.
  Use those wrappers only when those side effects are acceptable. Do not copy
  global shutdown behavior into new agent automation.
- `make client-stories` remains a useful local gallery, but currently pins a
  device/runtime and waits four seconds before capture. It is supplemental;
  use state-driven UI tests for repeatable interaction and screenshot evidence.

Run service changes through `PennyServicesStandaloneTests` and app/model changes
through `PennyClient`. The full CI-equivalent command is `make client-check`;
use the explicit `xcodebuild test` pattern with those schemes when selecting a
destination yourself. A docs-only change needs path/link and command verification,
not a device run. Report failures and skipped checks with their actual reasons.
