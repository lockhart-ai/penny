# PennyClient PR Review Guide

A focused checklist for reviewing Swift and iOS changes in `penny-client/`. Use it with the repository-wide PR review guide and [`penny-client/AGENTS.md`](../penny-client/AGENTS.md).

## 1. Use the framework as designed

This is the primary rule; every other rule in this guide sits under it.

- [ ] **Never rebuild an element iOS provides.** When SwiftUI has an element for the job, use it with its stock placement and behaviour: `TabView` (the bottom tab bar, `.badge`, the search role); `NavigationStack` with its visible bar, title, `.navigationSubtitle`, `.toolbar` items and the system back button; `List` with `Section`s; `Label`; `LabeledContent`; `.searchable`; `Menu` and `.contextMenu`; sheets; the system scroll-edge effect. A hand-built lookalike gets none of the system's glass, animation, accessibility or tap targets, and breaks in ways the stock element never does.
- [ ] **Compose, never fake.** Building our own element out of Apple's components and materials is fine (the live dock is a panel on real `.glassEffect`). What's never fine is faking what Apple renders or animates: hand-drawn glass or blur, a hand-rolled animation standing in for a system one (a geometry "morph" in place of the glass morph), painted fades over the system's scroll-edge effect, faked menus. If Apple provides the rendering or the animation, use theirs.
- [ ] **Never hide a system element to draw a lookalike over it.** A hidden navigation bar still takes touches in its region: the hand-built top tab capsule and glass circles, floated over hidden bars, shipped with dead taps (#3111).

## 2. Scope and architecture

- [ ] The diff is limited to the requested PennyClient behavior.
- [ ] Existing service, view-model, persistence, and protocol abstractions are reused instead of duplicated.
- [ ] Cross-stack changes identify the matching Python endpoint/model/test and preserve compatibility intentionally.
- [ ] No unrelated generated files, derived data, local secrets, or formatting churn are included.
- [ ] Target membership, build configurations, entitlements, and scheme changes are intentional.
- [ ] Service behavior and service-owned models live in `PennyServices`; SwiftUI presentation and view-model behavior remain in the app target.
- [ ] Package resources, including the Metal search shader, are loaded through the package boundary and do not rely on the app bundle.

## 3. SwiftUI and state

- [ ] View code remains primarily declarative; business logic lives in the view model or service layer.
- [ ] Observation state has one clear owner and does not create duplicate sources of truth.
- [ ] Async tasks are cancelled when their owning lifecycle ends.
- [ ] Loading, empty, error, retry, reconnect, and cancellation states are explicit.
- [ ] User actions are idempotent where repeated taps, reconnects, or callbacks are possible.
- [ ] Filtering, sorting, pagination, and “new messages” affordances preserve the reader's position.
- [ ] Main-actor isolation is correct; expensive SQLite, decoding, or network work is not accidentally performed on the main actor.

## 4. Concurrency and networking

- [ ] Websocket sends and receives are safe under reconnect and concurrent event delivery.
- [ ] Correlated responses cannot resume the wrong continuation or leave one hanging.
- [ ] Disconnects cancel or preserve pending work according to the intended resume behavior.
- [ ] Heartbeats, retries, timeouts, and backoff do not create duplicate requests or busy loops.
- [ ] Server errors and malformed payloads produce actionable local state without crashing the client.
- [ ] Request and response models use explicit coding keys and tolerate intentionally optional fields.
- [ ] Wire-format changes are tested in both directions, including missing optional fields and unknown fields where relevant.

## 5. Persistence and migrations

- [ ] All message/database access goes through `DatabaseService` or the established store abstraction.
- [ ] Database work uses isolated connections/queues appropriate to the service's concurrency model.
- [ ] Inserts, updates, deduplication, reconciliation, and attachment preservation are tested.
- [ ] Cursor and pagination ordering is stable and uses the intended `(createdAt, id)` tie-breaker.
- [ ] Local optimistic messages reconcile with canonical server messages without duplicates.
- [ ] Schema changes are backward-compatible or include the required migration and upgrade test.
- [ ] No user data is loaded or rewritten wholesale when a bounded query would suffice.

## 6. Security, privacy, and configuration

- [ ] Credentials, pairing tokens, device secrets, APNs tokens, and websocket auth are never logged or committed.
- [ ] `Prefs` and Keychain lookup precedence remains correct.
- [ ] Logs use `OSLogService` and privacy annotations; message content and attachment data remain private.
- [ ] Entitlements and bundle identifiers are correct for every affected target/configuration.
- [ ] Generated Info.plist values are verified in the built app when configuration behavior changes.
- [ ] Fixtures and test payloads contain synthetic data only.

## 7. UI and accessibility

- [ ] Existing navigation, toolbar, composer, attachment, and status conventions are preserved.
- [ ] Loading and error states are visible and recoverable.
- [ ] Dynamic type, VoiceOver labels, control roles, contrast, and hit targets remain usable.
- [ ] Buttons are not accidentally made borderless/plain or visually ambiguous.
- [ ] Layout is checked at narrow widths, large text sizes, empty states, and long content.

## 8. Tests and verification

- [ ] Focused tests cover the changed behavior and the most likely regression edge cases.
- [ ] View-model tests cover state derivation and side effects, not only initial rendering.
- [ ] Service tests cover reconnect, duplicate delivery, malformed input, and cancellation where applicable.
- [ ] `make client-services-check` passes for service or package changes and runs without building or launching PennyDev/PennyTestflight.
- [ ] `make client-check` also passes when the package public API, resources, persistence behavior, or app integration changes.
- [ ] Tests moved into `PennyServicesTests` do not import or depend on `PennyClient`; app-hosted tests import `PennyServices` directly when they exercise package types.
- [ ] Persistence tests use isolated/in-memory storage.
- [ ] The normal `make client-check` gate was run, or the PR explicitly states why it could not be run.
- [ ] If `DEVELOPER_DIR` selected Xcode beta, the exact Xcode version is recorded.
- [ ] A generic `build-for-testing` result is not reported as equivalent to running the simulator tests.
- [ ] SwiftLint completes with zero warnings and zero errors for every affected target/file; existing warnings in touched code are fixed rather than waived.
- [ ] Build warnings and SwiftLint output were reviewed; no warning is dismissed without an explicit project-level exception.

## 9. Hard-won patterns (Swift/iOS)

This list grows: whoever finds a new unhealthy pattern adds it here, in the same PR as the fix.

### Types and APIs

- [ ] **A component's action closure (`onSelect`, `onTap`, …) is a required parameter, never a default like `= { _ in }`.** A silent default lets a call site forget to wire the action while the view still compiles — the tab bar shipped with dead taps this way (#3105). Require the closure so a forgotten wiring is a compile error, not a shipped no-op.
- [ ] **No magic numbers in views or the kit.** Every size, gap, radius, and opacity is a named `Tokens` value (see `penny-client/DESIGN.md`) — a bare `16` or `0.6` drifts from the rest of the app the first time someone tweaks spacing elsewhere.
- [ ] **Screens pass data, not layout.** Rows are `ListRow` inside `SectionCard`; headers are `PageTitle`. A screen never sets row fonts, paddings, or alignment itself — a screen that sets its own fonts or paddings drifts out of alignment with every other screen.
- [ ] **Two views never claim one `glassEffectID` at once.** SwiftUI crashes when two views share an id simultaneously (#3097); use `matchedGeometryEffect` instead when a geometry morph between views is needed.
- [ ] **Display labels for tools are generic by default — never add tool-name special cases.** Tools are arbitrary and plugins add new ones at any time; a label keyed to a specific tool name silently stops working for every tool nobody enumerated.
- [ ] **New wire fields are optional on the Swift side.** An older server that hasn't shipped the field yet must still decode cleanly — make the field `nil`-able so a missing value decodes instead of throwing (#3047, #3051).
- [ ] **A `NavigationStack`'s toolbar is per screen, not inherited across a push.** Items a root view declares (a leading `Menu`, a trailing search item) don't carry onto a pushed `navigationDestination` unless that destination declares them again — and declaring a custom `.topBarLeading` item there *replaces* the automatic system back button rather than joining it (#3111). Shared chrome (a page switcher) belongs on the stack's root content only; a pushed page gets the system back button and whatever it declares itself.
- [ ] **A top-bar search needs `.searchToolbarBehavior(.minimize)` alongside `DefaultToolbarItem(kind: .search, …)`.** `.searchable` plus an explicit `DefaultToolbarItem(kind: .search, placement: .topBarTrailing)` renders the trailing icon, but without `.searchToolbarBehavior(.minimize)` tapping it blanks the whole bar instead of expanding a field (#3150) — confirmed in the component gallery (`SearchTopPage.swift`): all three of `.searchable`, `.searchToolbarBehavior(.minimize)`, and the explicit `DefaultToolbarItem` are required together on this SDK. Expanding presents as a floating glass capsule just above the keyboard, not a field docked into the title.

### Tests

- [ ] **A PR that adds or moves a control adds it to the UI test suite.** Add it to the XCUITest smoke suite (#3106), assert the state change it causes, and run `make client-ui-tests` locally before review. A story screenshot proves the look, never the behaviour.
- [ ] **Every search control gets a UI test that taps, types, and checks results.** Assert the field doesn't exist before the trailing icon is tapped, that typing changes what's visible (a row or run disappearing), and that Close removes the field again (#3150) — an icon that renders but does nothing on tap, or a filter that never filters, passes an existence-only check and still ships broken.
- [ ] **No fixed sleeps in tests.** Poll for the condition with a generous timeout instead of a fixed-duration sleep — a 1.7s sleep against a 1.5s grace window failed on CI (#3102) because CI timing isn't local timing.

### Text and layout

- [ ] **Styled text stays one paragraph.** Never render attributed-string runs as separate `Text` views inside a flow layout — merge adjacent non-link runs into one `Text`. Splitting them broke line wrapping: a highlight and bold text broke the sentence apart (#3098).
- [ ] **An inset's background continues the page background.** Content under a `safeAreaInset` or composer must not reveal the window behind it — a mismatch once left a visible seam above the composer (#3093).

### Stories and screenshots

- [ ] **Visual changes carry screenshots in the PR body.** Use the orphan `pr-media-<issue>` branch, halve the images, and place each next to its reference screen in `penny-client/Design/screens/` so a reviewer can compare without running the app.
- [ ] **Story fixtures look like production.** Every Penny reply carries her image, every bubble its metadata line, and the content is contrived but never assumes the user's pronouns — a story that looks unlike production hides the bugs production would show.
- [ ] **Story state that depends on async loading is applied after the load completes, not in `onAppear`.** Setting it in `onAppear` races the load, so the story renders the pre-load state instead of the one it was built to show (#3097).

### Process

- [ ] **Never hand-edit to debug.** A probe that replaces or comments out code to see what happens is reverted with `git restore`/`git checkout`, never retyped by hand — a hand "revert" once deleted the field's glass effect instead of restoring it (#3093).
- [ ] **Client checks that use the simulator take turns.** `make client-check`, `client-stories`, and `client-ui-tests` run under the shared simulator lock when agents work in parallel — two runs fighting over the same simulator both fail.

## 10. Review comments

Prioritize comments in this order:

- **P0:** data loss, credential/privacy exposure, corrupted persistence, or a release-blocking build failure.
- **P1:** broken user-visible behavior, protocol incompatibility, concurrency race, crash, or missing required migration.
- **P2:** regression risk, missing edge-case coverage, lifecycle leak, accessibility problem, or maintainability issue.
- **P3:** optional cleanup or style improvement that does not affect correctness.

Every actionable comment should identify the affected behavior, explain the failure mode, and point to a concrete correction or test.
