# Agent Task Workflow

The repeatable SOP a **task agent** follows to take **one** GitHub issue from assignment → merged PR → cleanup. One agent owns one ticket, in its own worktree, start to finish. This document is handed to the agent as its operating contract.

The golden rule underneath all of it: **stay in scope, keep the tree isolated, and never publish anything that isn't green and PII-clean.**

---

## 0. Inputs (what you're given)
- **One issue** — its number, scope, and any explicit "out of scope" boundary. Do not expand past it; if you discover adjacent work, note it for a follow-up issue, don't do it here.
- If the issue is `blocked` or `investigation`, **stop** — it needs a design pass, not implementation. Report back instead of coding.
- **Tickets rot — verify against the current code before implementing.** A ticket's file/line references and described structure were true when it was written; `main` may have moved since (a cited function refactored away, a mechanism reverted). Read the referenced code first and target *what exists now*, noting the discrepancy in your PR. If the drift is so large the ticket's premise no longer holds — or you hit a design decision the maintainer hasn't made — stop and report rather than guessing.

## 1. Branch in your own worktree (off clean main)
- Sync the baseline first: `git fetch origin main`. Branch from **`origin/main`**, never a stale local ref or another agent's branch.
- Work only in your **own** worktree. Never touch `main`'s working tree or another agent's worktree.
- Branch name is **descriptive**, not generic: `fix-timezone-local-render`, not `patch-1`.

## 2. Write a task plan file
- Create a short plan file in your worktree scratch area (gitignored / not part of the PR): the ticket link, your intended approach, the exact files you expect to touch, the **test strategy**, the **eval contract** if the change is model-facing, and the scope boundary you're holding.
- This is your working memory — update it as you go. It never lands in the PR.

## 3. Implement (to project convention)
- Follow the **Design Principles** in `CLAUDE.md` and the full **`docs/pr-review-guide.md`** rulebook: Pydantic for all structured data, short methods (10–20 lines), no magic strings, the database-stores pattern, FKs over denormalization, datetime columns for ordering, no silent fallbacks, no broad excepts.
- Prefer the **highest rung** that works: change behavior via data/prompt before code where the ticket allows.
- If `main` moves under you, **rebase** on it (`git rebase origin/main`) — resolve conflicts in place; never checkout/reset/branch-switch to escape a rebase.

## 4. Test — the one and only gate
- Run **exactly**: `make fix check 2>&1 | tee /tmp/check-output-$(git branch --show-current).txt; echo "EXIT_CODE=$pipestatus[1]" >> /tmp/check-output-$(git branch --show-current).txt`
  — the output path is **per-branch** because agents run concurrently: a shared `/tmp/check-output.txt` interleaves `EXIT_CODE` lines from sibling worktrees and makes a green line unattributable.
- Then read your branch's output file: check **`EXIT_CODE` first** (must be `0`), then grep for `FAILED` / `error[`.
- **If the run is interrupted** (e.g. `make: *** Error 130` from contention with a concurrent agent's Docker run), the output file is garbage — discard it and re-run the full gate cleanly. Never judge from a partial file.
- Never use `make pytest`, `make check` alone, or `docker compose run` directly.
- **Run the gate from *inside your worktree*, never the top-level checkout.** `make`/compose mount `./penny/penny` relative to the compose-file directory, so running `make` from the main repo dir tests *main*, not your branch — a green result there is meaningless. (This is separate from the §7 `make token` gotcha, which is the one thing you run against the main checkout.)
- **Build the image fresh so local tooling matches CI.** A cached local `penny` image can carry stale *pinned* tools (e.g. an old `ty`) and pass while CI's fresh build fails on the pinned version. `make build` before trusting the gate (`--no-cache` if a pin changed recently) so your local `ty`/`ruff` match CI's.
- **All code changes require tests.** Prefer folding assertions into an existing test over a new function; prefer integration tests through public entry points over unit tests.
- **Model-facing change?** (prompt / `extraction_prompt` / tool description / what the model reads) → it MUST land with a `tests/eval/` contract, and you must **dry-run it against the live model** (`make eval` / focused case) and read the result *before* committing. Validate each lever as you build it, not batched at the end.
  - **`make eval` self-serializes on the GPU — just run it.** The local GPU is single-tenant (only one eval can hold the model at a time), so the `eval` target queues via a **first-come-first-served ticket file** (`EVAL_QUEUE_DIR`): your invocation runs when its ticket is the oldest live one and the GPU is free, holds the ticket until the eval finishes, and while waiting prints your queue position and the current GPU holder. Tickets whose holder process died are reaped automatically, so a killed waiter can't wedge the line. Safe to run while sibling agents are active; no coordination on your part.
  - **Scope every eval run — NEVER the bare full suite.** The full suite is ~60 minutes of GPU time, and a queued full-suite run starves every sibling agent behind it. Always pass a focused subset: `EVAL_SAMPLES=3 make eval EVAL_PYTEST_ARGS="<your case file> -k <case> -v -m eval -s"` while iterating, and for the final non-regression pass name only the case files your change could plausibly affect. Run the bare full suite only when the user explicitly asks for it.
  - **Detach + record: the result log is the source of truth.** Run your eval so it survives you going dormant — a detached/background invocation teeing to a per-branch log (e.g. `/tmp/eval-<branch>.log`). On EVERY wake, **check that log before doing anything else**: the run may have completed while you slept even when no notification arrived. (Measured: three agents once had green eval results sitting in their logs all night while waiting for a wake-up that never came.)
  - **Self-heal on wake.** If your eval process/container no longer exists AND the log has no final summary line, the run died — relaunch it. Never conclude "still waiting" without confirming your waiter process actually exists. Your supervisor heartbeats the fleet (see `CLAUDE.md` → Agent Supervision), but the first line of defense is you checking your own artifacts on every wake.
  - **Capture the run report in the PR body.** Set `EVAL_REPORT_DIR=/penny/penny/tests/eval/_reports` on the `make eval` invocation (the target passes it through; the dir is mounted + gitignored, so the markdown survives the `--rm` container). The harness writes one `<case>.md` per case — a verbatim turn-by-turn transcript of every sample (👤 user · 🔧 Penny → tool · 📥 tool result · 🤖 Penny), with the structural verdict. Fold these into the **PR body**, one **`<details>` per case** (collapsed; the `<summary>` names the case + pass rate), under an always-visible summary table. This lets the reviewer read exactly what the model did on every run without a wall of text. The report is **never committed** — the PR body is the home; only the generator (in `tests/eval/conftest.py`) is committed.
- **penny-client (Swift) change?** → also run `make client-check` from your worktree: it builds the iOS app and runs `PennyClientTests` on a freshly booted simulator via `scripts/client-check.sh`. Requires Xcode — confirm with `xcodebuild -version` first. If Xcode is unavailable on your machine, push and let the `client-check` CI job be the gate, and **say so in the PR** ("Swift side verified by CI only") instead of claiming local verification. A simulator-infrastructure failure (`FBSOpenApplicationServiceErrorDomain` / "failed preflight checks") is a flake, not a code failure — the script's erase+boot hygiene prevents most of it; re-run before diagnosing your code.
- `EXIT_CODE=0` is a hard gate. Do not open a PR on red. For `penny-client/` diffs, a green `make client-check` (or, without Xcode, the green `client-check` CI check) is equally hard.

## 5. Quality review — before you publish
- With the test gate green, review your **full diff** against the project's canonical checklist *before* you commit or push. Invoke the **`/quality`** skill if it's available to you; otherwise read **`docs/pr-review-guide.md`** and self-review the diff against every applicable rule (error handling, forbidden patterns, async patterns, testing discipline, prompt engineering).
- Fix everything it surfaces. If you changed code, **re-run the §4 gate** (`EXIT_CODE=0`) before continuing.
- Don't push a diff you haven't run the checklist over.

## 6. Privacy gate — the repo is PUBLIC
- Before **any** commit or push, run the pre-publish PII checklist: no real user names, topics, dates, collections, handles, channel IDs, or run IDs in code, tests, fixtures, commit messages, or PR text. Genericize to synthetic equivalents.
- This is a hard line — it has been violated before. When in doubt, scrub.

## 7. Commit + open the PR
- `TOK=$(make token)` and **assert it's non-empty** before any `gh`/push — an empty token silently falls back to the wrong identity and creates PRs under the wrong author (immutable; must be closed + recreated).
- **Worktree gotcha:** a fresh worktree has no `.env` (it's gitignored), and Docker Compose creates a *directory* placeholder in its place — so `make token` fails inside the worktree with `failed to read .env: is a directory`. Run it against the primary checkout instead: `TOK=$(make -C <path-to-main-checkout> token)`. Token generation only reads config; it never touches that checkout's tree.
- **Push the branch first** (`GH_TOKEN=$TOK git push -u origin <branch>`), *then* `GH_TOKEN=$TOK gh pr create`.
- Commit message ends with the `Co-Authored-By:` trailer; PR body ends with the `🤖 Generated with Claude Code` trailer.
- PR body: what changed + why, the scope, **test evidence** (`EXIT_CODE=0`), eval results if applicable, and `Closes #<issue>`.

## 8. Shepherd the PR to merge (stay alive until MERGED)
You are **not done when the PR opens.** Do not exit — stay alive and shepherd the PR until it is **merged**, so a red CI or a moved `main` never sits unattended (that's exactly how a green PR silently rots). Loop:
- **Flag for the merge queue immediately after opening:** `gh pr merge <n> --auto` ("merge when ready") — do **not** pass a strategy flag: the merge queue sets the strategy and rejects `--squash` ("merge strategy is set by the merge queue"). The PR then enqueues itself the moment it is approved and green — no one has to click merge. If the queue later kicks it back out (its `merge_group` checks failed against latest `main`), treat that exactly like red CI: diagnose, fix, push. **A force-push clears the merge-when-ready flag** — re-run `gh pr merge <n> --auto` after every rebase push, and verify it took (`gh pr view <n> --json autoMergeRequest`).
- **CI:** poll `gh pr checks <n>`. If red, diagnose from the CI log (`gh run view <id> --log-failed`), fix, re-run the §4 gate (from your worktree), and push.
- **The merge queue owns serialization — do NOT proactively rebase every time `main` moves.** The queue tests every PR against latest `main` (plus anything queued ahead of it) via `merge_group` checks before merging, so "my branch is behind `main`" is not a problem you need to solve — and a push to an enqueued PR **dequeues it and restarts its checks**, so rebase-thrash actively delays your own merge. Rebase (`git fetch origin main && git rebase origin/main`, re-run the §4 gate, `git push --force-with-lease`) only when something is actually wrong: GitHub reports the PR has conflicts, the queue bounced your PR, or a change on `main` broke your branch semantically (e.g. a signature change in code you call).
- **Review:** address every review comment; re-gate; push.
- **Never destructively escape a rebase;** resolve conflicts in place. Re-verify the PR is still open before each push — if it merged, stop and go to §9.
- When CI is green, the PR has no conflicts, there are no open review threads, and the merge-when-ready flag is set: report "green · queued for merge (awaiting approval)" and **pause** — you'll be resumed on new CI / review / `main` activity. Don't busy-spin; pause between cycles.
- **Pause only on a RESOLVED state — never while a check is pending.** Ending your turn "to wait for CI" on the strength of a background monitor is the classic dead-waiter: unless you have **verified** a live waiter process exists, nothing will ever wake you (four agents in one fleet parked this way on phantom waiters). While any check is pending, poll `gh pr checks` in-turn with short sleeps until it resolves; only the "green · queued for merge" state (or an explicit supervisor-acknowledged block) is a legitimate place to stop.

## 9. Terminal cleanup — the REQUIRED last step of every task
- The **user** approves (code-owner review — no self-approval); the **merge queue** merges. With the merge-when-ready flag set (§8), approval alone carries the PR through the queue to merged with no further action from anyone.
- **Cleanup is not optional and not someone else's job — it is the final step of YOUR task, and your task is not complete until it's done.** The trigger is your PR reaching a **terminal state, either way**:
  - **Merged** → clean up.
  - **Closed without merge** (superseded, deferred, rejected) → clean up all the same. A closed PR's worktree is exactly as dead as a merged one's; this is the case that orphans trees.
- Cleanup = **first, from inside your worktree, run `make clean-project-images`** to drop the project-scoped `agent-<hash>-team` image your `make fix check` built (it must run while you're still in the tree, since it resolves your compose project by directory) — it is never reused once your worktree is gone, and left unswept these pile up until the disk fills (56 once took it to 99%). Then remove your worktree (`git worktree remove <path>`), delete the local **and** remote branch (skip the remote if GitHub already deleted it on merge), and discard the task plan file. Keep everything in place until the terminal state; nothing in place after it.
- Each pause-cycle while shepherding (§8), re-check the PR's state first — if it went terminal while you slept, run cleanup **now**, report "merged → cleaned up" (or "closed → cleaned up"), and end. Never end your final turn with the worktree still in place.
- If the change warrants it, update `CLAUDE.md` / `README.md` (docs-maintenance rule) — as part of the PR, not after.

---

## Invariants (true at every step)
1. **One ticket per agent; hold the scope boundary.** Adjacent work → a new issue, not this PR.
2. **Isolated worktree, branched from `origin/main`.** Never main's tree, never another agent's.
3. **`make token` non-empty check** before every GitHub op.
4. **`make fix check` is the only test path; `EXIT_CODE=0` is the gate** — run it *from your worktree* on a *freshly built* image, to your branch's own output file, re-run cleanly if interrupted.
5. **Quality-review the diff** against `docs/pr-review-guide.md` (or `/quality`) before publishing.
6. **PII pre-publish check** before anything leaves the machine.
7. **Model-facing change ⇒ committed `tests/eval/` contract, dry-run first.**
8. **Flag merge-when-ready (`gh pr merge <n> --auto` — no strategy flag; re-flag after every force-push); the merge queue owns serialization** — rebase only on conflicts or a queue bounce, and never destructively escape a rebase.
9. **Stay alive shepherding the PR** — CI green, reviews addressed, queued — until it reaches a **terminal state (merged OR closed)**; then **cleanup is the required last step** — worktree, branches, plan file — before you end.
