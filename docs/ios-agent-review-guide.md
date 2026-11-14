# iOS agent review guide

Use this for changes to `penny-client/` and its build/test scripts. Read the diff
and relevant surrounding code, then apply six checks:

1. **Correctness:** Does the changed flow meet the request? Check the relevant
   empty, loading, error, disconnected and cancellation paths. Look for stale
   asynchronous updates, duplicate sends and lost user state.
2. **Design:** Follow [DESIGN.md](../penny-client/DESIGN.md): native controls,
   recognizable SF Symbols, useful accessibility labels and no unnecessary copy.
   Check the relevant [UI regression rules](ios-agent-workflow.md#ui-regression-guidance),
   especially action wiring, navigation, glass identity, text wrapping and search.
3. **Device support:** Verify iPhone and iPad behavior, including narrow windows,
   rotation, large text and the keyboard when relevant. Review local screenshots
   for visual changes; note missing device coverage explicitly.
4. **Boundaries and data:** Keep presentation, services and persistence in their
   existing layers. Check protocol compatibility and protect saved messages,
   credentials and settings. Apply the [logging conventions](../penny-client/AGENTS.md#logging-conventions):
   existing logging abstraction, private defaults, no sensitive payloads and safe
   debug timings. Keep fixture hooks Debug-only.
5. **Evidence:** Require focused tests for changed logic and local UI tests with
   screenshots for changed interactions. Keep UI tests/capture out of CI; retain
   unit/service coverage. Require strict SwiftLint with zero violations and
   build/test logs showing zero warnings for Swift or build changes; a green exit
   code alone is insufficient. Documentation-only edits need accurate commands
   and links.
6. **Scope:** Avoid unrelated refactors, dependency/signing changes, generated
   artifacts and speculative abstractions. Keep affected docs accurate.

Report actionable findings in severity order with a file/line, concrete trigger,
user impact and suggested correction. Separate blockers from optional suggestions;
skip personal style preferences. If there are no findings, say so and state what
was tested and what remains unverified. Do not claim an unrun check passed.

See the [workflow](ios-agent-workflow.md) for destinations, commands and evidence.
