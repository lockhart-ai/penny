# Agent Task Workflow

The repeatable SOP a **task agent** follows to take **one** GitHub issue from assignment → merged PR → cleanup. One agent owns one ticket, in its own worktree, start to finish. This document is handed to the agent as its operating contract.

The golden rule underneath all of it: **stay in scope, keep the tree isolated, and never publish anything that isn't green and PII-clean.**

---

## 0. Inputs (what you're given)
- **One issue** — its number, scope, and any explicit "out of scope" boundary. Do not expand past it; if you discover adjacent work, note it for a follow-up issue, don't do it here.
- If the issue is `blocked` or `investigation`, **stop** — it needs a design pass, not implementation. Report back instead of coding.

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
- **All code changes require tests.** Prefer folding assertions into an existing test over a new function; prefer integration tests through public entry points over unit tests.
- **Model-facing change?** (prompt / `extraction_prompt` / tool description / what the model reads) → it MUST land with a `tests/eval/` contract, and you must **dry-run it against the live model** (`make eval` / focused case) and read the result *before* committing. Validate each lever as you build it, not batched at the end.
- `EXIT_CODE=0` is a hard gate. Do not open a PR on red.

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

## 8. Address review feedback
- **Before every push**, verify the PR is still open (`gh pr list --head <branch>`). If it's merged, stop — start a fresh branch for follow-ups, don't push to the merged one.
- Rebase on latest `main` as needed (in place — no destructive escape).
- Re-run the **§4** gate after every change; `EXIT_CODE=0` before pushing.

## 9. Merge → cleanup
- The **user** merges (branch protection: no self-merge to `main`).
- After merge: remove your worktree (`git worktree remove …`), delete the local and remote branch, and discard the task plan file.
- If the change warrants it, update `CLAUDE.md` / `README.md` (docs-maintenance rule) — as part of the PR, not after.

---

## Invariants (true at every step)
1. **One ticket per agent; hold the scope boundary.** Adjacent work → a new issue, not this PR.
2. **Isolated worktree, branched from `origin/main`.** Never main's tree, never another agent's.
3. **`make token` non-empty check** before every GitHub op.
4. **`make fix check` is the only test path; `EXIT_CODE=0` is the gate** — written to your branch's own output file, re-run cleanly if interrupted.
5. **Quality-review the diff** against `docs/pr-review-guide.md` (or `/quality`) before publishing.
6. **PII pre-publish check** before anything leaves the machine.
7. **Model-facing change ⇒ committed `tests/eval/` contract, dry-run first.**
8. **Rebase, don't destructively escape;** commit before any branch/rebase probing.
9. **Green + reviewed + user-merged** — then, and only then, clean up.
