# PennyClient PR Review Guide

A focused checklist for reviewing Swift and iOS changes in `penny-client/`. Use it with the repository-wide PR review guide and [`penny-client/AGENTS.md`](../penny-client/AGENTS.md).

## 1. Scope and architecture

- [ ] The diff is limited to the requested PennyClient behavior.
- [ ] Existing service, view-model, persistence, and protocol abstractions are reused instead of duplicated.
- [ ] Cross-stack changes identify the matching Python endpoint/model/test and preserve compatibility intentionally.
- [ ] No unrelated generated files, derived data, local secrets, or formatting churn are included.
- [ ] Target membership, build configurations, entitlements, and scheme changes are intentional.

## 2. SwiftUI and state

- [ ] View code remains primarily declarative; business logic lives in the view model or service layer.
- [ ] Observation state has one clear owner and does not create duplicate sources of truth.
- [ ] Async tasks are cancelled when their owning lifecycle ends.
- [ ] Loading, empty, error, retry, reconnect, and cancellation states are explicit.
- [ ] User actions are idempotent where repeated taps, reconnects, or callbacks are possible.
- [ ] Filtering, sorting, pagination, and “new messages” affordances preserve the reader's position.
- [ ] Main-actor isolation is correct; expensive SQLite, decoding, or network work is not accidentally performed on the main actor.

## 3. Concurrency and networking

- [ ] Websocket sends and receives are safe under reconnect and concurrent event delivery.
- [ ] Correlated responses cannot resume the wrong continuation or leave one hanging.
- [ ] Disconnects cancel or preserve pending work according to the intended resume behavior.
- [ ] Heartbeats, retries, timeouts, and backoff do not create duplicate requests or busy loops.
- [ ] Server errors and malformed payloads produce actionable local state without crashing the client.
- [ ] Request and response models use explicit coding keys and tolerate intentionally optional fields.
- [ ] Wire-format changes are tested in both directions, including missing optional fields and unknown fields where relevant.

## 4. Persistence and migrations

- [ ] All message/database access goes through `DatabaseService` or the established store abstraction.
- [ ] Database work uses isolated connections/queues appropriate to the service's concurrency model.
- [ ] Inserts, updates, deduplication, reconciliation, and attachment preservation are tested.
- [ ] Cursor and pagination ordering is stable and uses the intended `(createdAt, id)` tie-breaker.
- [ ] Local optimistic messages reconcile with canonical server messages without duplicates.
- [ ] Schema changes are backward-compatible or include the required migration and upgrade test.
- [ ] No user data is loaded or rewritten wholesale when a bounded query would suffice.

## 5. Security, privacy, and configuration

- [ ] Credentials, pairing tokens, device secrets, APNs tokens, and websocket auth are never logged or committed.
- [ ] `Prefs` and Keychain lookup precedence remains correct.
- [ ] Logs use `OSLogService` and privacy annotations; message content and attachment data remain private.
- [ ] Entitlements and bundle identifiers are correct for every affected target/configuration.
- [ ] Generated Info.plist values are verified in the built app when configuration behavior changes.
- [ ] Fixtures and test payloads contain synthetic data only.

## 6. UI and accessibility

- [ ] Existing navigation, toolbar, composer, attachment, and status conventions are preserved.
- [ ] Loading and error states are visible and recoverable.
- [ ] Dynamic type, VoiceOver labels, control roles, contrast, and hit targets remain usable.
- [ ] Buttons are not accidentally made borderless/plain or visually ambiguous.
- [ ] Layout is checked at narrow widths, large text sizes, empty states, and long content.

## 7. Tests and verification

- [ ] Focused tests cover the changed behavior and the most likely regression edge cases.
- [ ] View-model tests cover state derivation and side effects, not only initial rendering.
- [ ] Service tests cover reconnect, duplicate delivery, malformed input, and cancellation where applicable.
- [ ] Persistence tests use isolated/in-memory storage.
- [ ] The normal `make client-check` gate was run, or the PR explicitly states why it could not be run.
- [ ] If `DEVELOPER_DIR` selected Xcode beta, the exact Xcode version is recorded.
- [ ] A generic `build-for-testing` result is not reported as equivalent to running the simulator tests.
- [ ] SwiftLint completes with zero warnings and zero errors for every affected target/file; existing warnings in touched code are fixed rather than waived.
- [ ] Build warnings and SwiftLint output were reviewed; no warning is dismissed without an explicit project-level exception.

## 8. Review comments

Prioritize comments in this order:

- **P0:** data loss, credential/privacy exposure, corrupted persistence, or a release-blocking build failure.
- **P1:** broken user-visible behavior, protocol incompatibility, concurrency race, crash, or missing required migration.
- **P2:** regression risk, missing edge-case coverage, lifecycle leak, accessibility problem, or maintainability issue.
- **P3:** optional cleanup or style improvement that does not affect correctness.

Every actionable comment should identify the affected behavior, explain the failure mode, and point to a concrete correction or test.
