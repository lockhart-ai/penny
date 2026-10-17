# Agent Task Workflow (the kitten SOP)

The repeatable SOP a **kitten** follows to take **one** GitHub issue from a brief to a reviewed, green draft PR. A kitten is a Sonnet, Haiku, or GLM subagent a Glade task's **coordinator** (Opus) starts with the `Agent` tool, its model always set explicitly. The coordinator plans, dispatches, reviews, marks a PR ready, and merges — it never implements, gates, or runs an eval itself. This document is the kitten's operating contract; the coordinator's own duties are `docs/agent-supervisor-runbook.md`.

The golden rule underneath all of it: **stay in scope, keep the tree isolated, and never publish anything that isn't green and PII-clean.**

---

## 0. The hard lines

- **You never end a turn holding uncommitted edits.** Once you've edited, commit and push before you stop or report — even when you're stopping to ask a question — so the work exists on the remote, not only in a worktree. (A plan-only turn, with no edits, has nothing to push.)
- **A plan-first brief means you cut a named worktree yourself, as your first step** (§1) — a harness-made one is removed when a turn ends without changes, and a removed worktree is work that no longer exists anywhere.
- **You never start subagents of your own.** One kitten, one issue, no sub-dispatch.
- **Hold your brief's bound.** When you hit its stop rule or its named file/scope limit, stop and report — never widen the job to cover something adjacent. Anything beyond a small edit is plan-first: return a short plan (files, approach, size) and edit only once the coordinator approves it.
- **Never yours to do unprompted:** anything touching production (`make prod`/`up`/`kill`), the live database (including `make migrate-test`, which reads a copy of it), or anything outward outside the repo. Put the need in your report instead.
- **You never contact the code owner or the user.** A question or a judgment call goes in your report to the coordinator — that's the only channel you have.
- **You never mark a PR ready, flag merge-when-ready, or approve** — on any change, however small. The coordinator does that after its own review.

## 1. Get into your worktree

Which applies depends on what your brief says:

- **Your brief says you're already in a worktree and that you'll edit in your first turn.** Stay in it — never run `git worktree add` yourself; a second worktree on top of the one you were launched into is always wrong.
- **Your brief is plan-first (a plan this turn, edits only after approval).** A harness-provisioned worktree is removed automatically when the turn that used it ends with no changes, so a plan-only turn loses it. Cut your own **named** one instead, as your first step, from the primary checkout: `git fetch origin main`, then `git worktree add -b <branch> /Users/decker/Documents/penny/.claude/worktrees/<name> origin/main`. A worktree you cut this way persists across turns.

Either way: work only inside your worktree, with absolute paths (your shell's cwd does not reliably persist between commands). Replace the placeholder `.env` with a symlink to the primary checkout's before anything mints a token: `[ -L .env ]` is the check (Docker's placeholder is an empty *directory*, which `[ -e .env ]` would wrongly pass) — `ln -sfn /Users/decker/Documents/penny/.env .env`. **Never open `.env`.** For a plain token mint, `make -C /Users/decker/Documents/penny -s token | tail -1` works from anywhere; with the symlink in place, `make token` also works from inside your own tree, which anything whose code must be your branch's (`make eval-report`) requires.

## 2. Read your brief and the issue

- Read the issue, its scope, and any explicit "out of scope" boundary before touching anything.
- If it's `blocked` or `investigation`, **stop** — it needs a design pass, not implementation. Report back instead of coding.
- **Tickets rot.** A ticket's file/line references were true when written; `main` may have moved. Read the referenced code first and target what exists now, noting the discrepancy in your PR. If the drift is large enough that the ticket's premise no longer holds, stop and report rather than guessing.

## 3. Implement (to project convention)

- Follow `CLAUDE.md`'s Design Principles and the full `docs/pr-review-guide.md` rulebook. Prefer the highest rung that works: behavior via data/prompt before code, where the ticket allows.
- If `main` moves under you, rebase (`git rebase origin/main`) — resolve conflicts in place; never checkout/reset/branch-switch to escape one.
- A **DB migration** gets its number from `origin/main`'s migrations directory, picked after your fetch, never from the ticket text — `main` moves, and a duplicate prefix collides at merge. Re-check the number on every rebase; `make migrate-validate` is the tripwire.

## 4. Docs land in the same PR

Update `CLAUDE.md` / `README.md` / the relevant doc as part of this PR, not a follow-up, whenever the change warrants it.

## 5. Quality review — before the gate

Review your full diff against `docs/pr-review-guide.md` (the `/quality` skill, if available, does this). Fix everything it surfaces, then gate the finished diff once — quality-before-gate is what makes a single gate run possible.

## 6. Privacy gate — the repo is PUBLIC

Before any commit or push: no real user names, topics, dates, collections, handles, channel IDs, or run IDs in code, tests, fixtures, commit messages, or PR text. This is a hard line. When in doubt, scrub.

## 7. The gate

- **A docs-only diff runs no local gate at all** — nothing in `make fix check` validates markdown, and CI's required check covers branch protection.
- Otherwise, run **exactly** `make fix check 2>&1 | tee /tmp/check-output-$(git branch --show-current).txt; echo "EXIT_CODE=$pipestatus[1]" >> /tmp/check-output-$(git branch --show-current).txt` **once**, at the end, from inside your worktree — never `make pytest`, `make check` alone, or `docker compose run` directly. If your isolation guard refuses the chained form, run the `tee` pipeline alone, then a plain `make fix check` for the exit status. Read `EXIT_CODE` first, then grep for `FAILED` / `error[`. A run interrupted by contention is garbage — discard it and re-run clean.
- `penny-client/` changes additionally need `make client-check` (requires Xcode — `xcodebuild -version` first; without it, push and say so, relying on the `client-check` CI job).
- All code changes carry tests: fold assertions into an existing test over a new one, integration tests over unit tests where possible, and a render that enters the model's context is asserted **whole**, never by substring (`pr-review-guide.md` → Whole-Render Assertions).
- `EXIT_CODE=0` (or green `client-check`) is a hard gate. Do not open a PR on red.

## 8. Model-facing and eval-case changes — your own scoped eval

A change to a prompt, `extraction_prompt`, tool description, or anything else the model reads — **or a PR that adds or changes an eval case** — needs a live-model result before you commit, and the suite's model carries that result, never an assumption:

- `LLM_MODEL=google/gemma-4-26b-a4b-it make eval-remote EVAL_LEVER="<one line on what changed>" EVAL_PYTEST_ARGS="<your case's test paths> -m eval -s"` — scoped to your own cases, never the bare full suite (that needs the coordinator's and code owner's explicit go-ahead).
- Read the result before committing. Post it with `make eval-report PR=<n>`, run from your own worktree, **after** the PR exists.
- Each re-run (yours, or one the coordinator asks for) adds a **new** PR comment — never rewrite the body or overwrite a prior comment; the comment stream is the durable record.
- If your ticket is one round of a larger beat, `docs/eval-iteration-workflow.md` is the loop it's a round of.
- If your brief explicitly defers eval validation to an end-of-epic sweep, skip the dry-run but add a "Model-facing changes" section to the PR body inventorying exactly what changed.
- **Writing a case** (new or ported): [`docs/eval-working-guide.md`](eval-working-guide.md) §8 "Building the world and the five wordings" is how to build the fixture; §10 "Reading a run" is how to read what comes back.
- **A failure-driven case** (your issue carries an agreed case proposal, not just a prompt change) follows its own order, all on the one draft PR: open the PR with the case as written; run it scoped and post the result as the "before" (`EVAL_LEVER="reproduces: <the failure>"`); push the fix as new commits on the same PR; run it scoped again and post the "after" (`EVAL_LEVER="fix: <the lever>"`). [`docs/eval-working-guide.md`](eval-working-guide.md) §1 is the whole path.

## 9. Commit, push, open a draft PR

- Mint the token inline, immediately before the call it authorizes, and assert it's non-empty — never stash one in a file: `GH_TOKEN=$(make -C /Users/decker/Documents/penny -s token | tail -1) gh …` (or the symlinked in-tree form). An empty token is a stop, not a fallback to ambient `gh` auth.
- Immediately before pushing, `git fetch origin main` and re-check whether it moved; rebase first if it plausibly touches your files.
- Push the branch, then `gh pr create --draft`. Title is `type(scope): title` (six types — `feat`/`fix`/`test`/`refactor`/`chore`/`docs`; scopes `penny`/`eval`/`harness`/`client`/`browser`/`ci` — per `CLAUDE.md` → Git Workflow). Body opens with one sentence that summarises the whole PR on its own — what changes and why, readable cold by someone who hasn't seen the issue (the code owner reads PRs as phone notifications showing only that line) — then `Because` / `This commit`, ends with `Closes #<issue>`, no generated-with footer, no session URL.
- Right after creating it, read back `gh pr view <n> --json mergeable,autoMergeRequest`. If `CONFLICTING`, fix it now.

## 10. Wait for CI

Poll `gh pr checks <n>` in-turn until it resolves. Red → diagnose from the CI log and push a fix. Never flag merge-when-ready, mark the PR ready, or approve — that's the coordinator's, once its own review is done.

## 11. Report back

PR link; how each acceptance point was checked; the eval run posted (if any) and its result; decisions you made where the issue was silent; open questions. **Quote, don't describe:** the gate's last line (`EXIT_CODE=…`, read from your output file) and the output of `git log --oneline -2` taken right after your push, both pasted verbatim — a description of a green run is not evidence of one. Leave your worktree and branch exactly as they are — **cleanup is not yours**; the coordinator clears them once the PR reaches a terminal state (merged or closed).

## 12. Review fixes

When the coordinator resumes you with changes: make them as new commits on the **same** branch — no force-push unless a rebase made one necessary. Re-run the §7 gate if anything but docs changed. For a model-facing or eval-case change, re-run the scoped eval at the new head and post it; a rebase that leaves the tree byte-identical to the measured head (`git rev-parse HEAD^{tree}` equal on both) keeps the prior run valid — say so in a PR comment instead of re-running. Then report again, the same shape as step 11.

---

## Invariants (true at every step)

1. **One issue per kitten; hold the brief's scope and stop rule.** No sub-dispatch; adjacent work is a question for the coordinator, not this PR.
2. **Worktree discipline**: already in one and edit in the first turn → stay; plan-first → cut a named one yourself, else the harness reclaims it. Never two.
3. **Token guard**: minted inline, never stashed; empty is a stop. Never open `.env`.
4. **`make fix check` is the only local gate** (docs-only diffs run none), once at the end, `EXIT_CODE=0` hard.
5. **Quality-review before the gate; PII-scrub before anything leaves the machine.**
6. **A model-facing or eval-case change gets a scoped `make eval-remote` run on the suite's model, posted to the PR**, before you commit and again after a review-fix round (unless the tree is byte-identical to a prior measured head).
7. **Production, the live DB, and anything outward are never unprompted** — they go in the report as a need.
8. **You never mark a PR ready, flag merge-when-ready, or approve.** You stop at a green, reviewed draft PR and report; the coordinator takes it from there, including cleanup.
9. **You never end a turn with an edit uncommitted or a commit unpushed** — not even to stop and ask a question. Your report quotes the gate's `EXIT_CODE` line and `git log --oneline -2` verbatim, never a description of them.
