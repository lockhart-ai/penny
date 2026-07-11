# PennyClient Agent Task Workflow

The repeatable SOP for taking one PennyClient task from issue to merged PR and cleanup. This workflow supplements [`penny-client/AGENTS.md`](../penny-client/AGENTS.md) and the repository-wide [`docs/agent-task-workflow.md`](agent-task-workflow.md).

The operating rule is: keep the change narrow, preserve the client/server contract, verify the real iOS build and tests, and do not publish a PR with unreviewed warnings or simulator-only assumptions.

## 0. Establish scope

- Read the issue, its comments, and the current `penny-client/` checkout before coding.
- Confirm whether the task is UI, view-model, persistence, websocket/protocol, push notification, signing, or cross-stack work.
- Identify the matching Python/server contract before changing Swift wire models.
- Record explicit out-of-scope items. Do not silently expand into backend changes, migrations, or signing changes.
- If the issue is primarily exploratory or requires a product decision, report the findings before implementing.

## 1. Isolate the task

- Work in an isolated worktree branched from the current `origin/main`, following the repository workflow.
- Check `git status` before editing. Preserve unrelated user changes and do not reset or discard them.
- Create a short scratch plan containing the issue, scope boundary, expected files, test strategy, and any simulator/Xcode assumptions.

## 2. Inspect the existing client path

Trace the behavior before editing:

1. View or view-model entry point.
2. Service or websocket request/response path.
3. Persistence path through `DatabaseService`.
4. State restoration, preferences, and reconnect behavior.
5. Existing focused tests and test doubles.
6. Server-side protocol models/tests when the task crosses the websocket boundary.

Prefer the existing Observation, SQLite.swift, `Prefs`, `OSLogService`, and websocket abstractions. Avoid introducing a parallel service or a second representation of server data.

## 3. Implement narrowly

- Keep SwiftUI rendering, state transitions, persistence, transport, and protocol encoding in their established layers.
- Use explicit dependencies and test doubles; do not rely on live network state or the user's persistent database in tests.
- Treat generated Info.plists, entitlements, build settings, and target membership as part of the behavior when the task touches app configuration.
- Preserve protocol compatibility deliberately. Update Python models, Swift wire models, documentation, and both sides' tests together when a payload changes.
- Use `LogService`/`OSLogService` with privacy-safe dynamic values. Never log message bodies, credentials, tokens, secrets, attachment contents, or raw SQL containing user data.
- Do not change SwiftLint rules without explicitly asking the user first; failure to do so is punishable by a lightcycles deathmatch.

## 4. Test while implementing

- Add or update focused Swift Testing tests for every changed state transition or service behavior.
- For view-model changes, cover derived state, user actions, validation, filtering/sorting, and async lifecycle behavior as applicable.
- For persistence changes, use isolated/in-memory database setup and test inserts, updates, deduplication, migration behavior, and attachment preservation.
- For websocket changes, assert encoded requests, decoded responses, reconnect/resume behavior, duplicate delivery, and error handling.
- For cross-stack changes, run the focused Python protocol/channel tests as well as the client tests.

## 5. Run the gates

From the repository root, the normal PennyClient gate is:

```bash
make client-check
```

This builds the client and runs `PennyClientTests` on a freshly booted simulator. Treat build errors, SwiftLint warnings, test failures, signing/configuration errors, and simulator failures separately in the report.

When Xcode exposes a connected physical iOS device as a valid run target, prefer it over a simulator for manual verification and device-specific behavior. Use the physical device for push notifications, Keychain/device identity, networking, camera/media, performance, and other hardware-dependent checks. Confirm the target with Xcode's available destinations or `xcodebuild -showdestinations` before running. A physical-device failure caused by signing, provisioning, developer mode, trust, or device availability is an environment gate; report it separately from source/build failures and fall back to the simulator when appropriate.

The preferred target order is:

1. Connected physical device, when available and provisioned.
2. iOS Simulator, for repeatable automated tests and when no usable device is available.
3. Generic `build-for-testing`, only for compile/toolchain verification when neither a device nor simulator test run is possible.

When multiple Xcode installations exist, verify the selected toolchain first:

```bash
xcode-select -p
xcodebuild -version
```

To use Xcode beta for a single invocation:

```bash
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer make client-check
```

If simulator services are unavailable but compilation needs verification, use a generic build-for-testing invocation with the same `DEVELOPER_DIR`. A successful generic build proves compilation and toolchain use, but does not replace the full simulator test gate.

Do not claim tests passed when only a generic build or a partial compile completed.

## 6. Review the diff

Before committing, run the checklist in [`penny-client-pr-review-guide.md`](penny-client-pr-review-guide.md). At minimum verify:

- no unrelated files or generated artifacts;
- no new warnings or SwiftLint violations;
- no duplicated protocol transformations or constants;
- no unsafe concurrency or main-actor violations;
- persistence and reconnect behavior remain correct;
- tests cover the changed behavior and important edge cases;
- no secrets or personal data entered fixtures, logs, comments, or PR text.

Re-run the relevant gate after any code change made during review.

## 7. Publish and shepherd

Follow the repository-wide PR process for commit, push, CI, review comments, merge queue, and PII checks. The PR body should include:

- the user-visible and internal behavior change;
- the protocol/server impact, if any;
- the exact client gate and Xcode version used;
- any simulator or signing limitation;
- focused test evidence;
- a clear note when local verification was build-only and CI remains the test gate.

Stay with the PR through CI and review. A PennyClient PR is not complete while checks are red, review threads are unresolved, or a terminal PR's worktree remains orphaned.

## Invariants

1. One task, one scope boundary, one isolated worktree.
2. The server/client protocol has one intentional source of truth.
3. User-visible state is testable without live services.
4. Local build success and simulator test success are reported separately.
5. Warnings, privacy leaks, and configuration drift are release-blocking issues.
