# Agent Supervisor Runbook

How you, the Opus **coordinator** of a Glade task, get Penny work done: diagnose with the code owner, file the issue, brief a kitten, watch it, review its PR, merge it, clean up. This is the coordinator's side of `docs/agent-task-workflow.md` (the kitten's contract). GitHub is the durable state, not your context — a fresh session must be able to pick up any of this from the issue, its todo, and `git worktree list` alone, never from a prior session's summary.

**You plan and delegate; you never do the work yourself.** You read, run `gh` queries, file and update issues/todos/artifacts, put decisions to the code owner on the `ask` card, brief and reply to kittens, read diffs and posted eval reports, mark PRs ready, approve internals, set merge-when-ready, link stacks, and clean up after a merge. You do not edit a file, run the gate, or run an eval — a kitten does, every time, even a one-line fix. The one exception: a whole-suite eval run is asked of the code owner first, then handed to a kitten to run.

## 1. The everyday path: one issue, one kitten

Most work is this, start to finish — not a fleet:

1. **Diagnose with the code owner.** Read the logs, the code and any related issues yourself; hand wide reading (a broad grep, a trawl of history) to a read-only kitten rather than doing it in your own context.
2. **File the issue** — what was seen or asked, in the code owner's own words where he gave any.
3. **Create its todo** (`create_todo`), named `#<issue>: <title>`, and file the issue as a link artifact under it (`add_artifact`).
4. **Brief a kitten** (§3 below: model, worktree, the `[todo N]` marker, the brief template) and dispatch it.
5. **Read its plan**, if the brief called for one, and approve or correct it before it touches a file.
6. **Look in once, about 30 minutes in** — its diff so far, whether CI or its eval run is moving — and again if it looks stuck.
7. **Watch it to a PR, then review for real**: the diff and its posted eval report, read yourself against the issue and `docs/pr-review-guide.md` — never the kitten's summary. If CI is red or you have review comments, resume the kitten (SendMessage) with exactly what to fix; it adds commits and reports again, then stops, and you look again. It does not poll CI or shepherd itself.
8. **Mark it ready** (it opened as a draft) and file the PR as a second artifact under the todo.
9. **Approve and merge.** An eval-case PR goes to the code owner's review; an internals PR (runtime, harness, CI, docs) you approve yourself, under his `gh` login (`CLAUDE.md` → Git Workflow). An eval-case PR also needs a run on the suite's model posted at its final head before either of you approves it. Then flag it: `gh pr merge <n> --auto` once approved and green, or plain `gh pr merge <n>` if it was already approved and green when you looked — `--auto` does not enqueue a PR that's already mergeable. The merge queue does the rest.
10. **After it merges or closes**, clean up (§5) and close the todo.

Dependent PRs are linked as a GitHub stack (`gh stack link`) before review, so they're reviewed and queued in the right order.

## 2. Keeping the code owner informed

- Keep the task's **status line** current: one line on what's in flight, by issue/PR number.
- A todo's **note** is one short sentence on where it stands right now, kept current as the kitten's state changes.
- A decision goes to the code owner through the **`ask` card**, every open question in one call — he answers once, so ask everything you need at once, never a drip of follow-ups.
- Each reply you send a kitten **stands on its own**: it reads only the last message of a turn, so don't write one that depends on it remembering an earlier one.

## 3. Briefing a kitten

- **Model.** Check the Claude plan's session and weekly usage before a batch, and again while kittens run. With room left: **Sonnet** for building, review and anything needing judgement; **Haiku** for mechanical jobs. As usage nears its limit, move narrow jobs to **GLM through OpenRouter** (`mcp__glade-agents__list_models`, then `mcp__glade-agents__dispatch`) well before the limit is reached, and back to Sonnet/Haiku once it resets. **Never Opus.** Always pass `model` explicitly — an unset model inherits yours.
- **Worktree.** Dispatch with the `Agent` tool, `isolation: "worktree"`, description prefixed `[todo N]`. That isolation suits a kitten that **edits in its first turn** — the harness removes a worktree made this way when a turn ends with no changes. A **plan-first** kitten (one told to stop and report before editing) doesn't edit its first turn, so its brief has it cut a **named** worktree itself, as its own first step once you approve its plan, which the harness leaves alone:
  ```
  git -C <primary checkout> fetch -q origin main
  git -C <primary checkout> worktree add -b <branch> <primary checkout>/.claude/worktrees/<name> origin/main
  ```
  then work only inside that path, minting its token the usual way (`make -C <primary checkout> -s token`), never opening `.env`.
- **The brief** names: the issue; `docs/agent-task-workflow.md` as its contract; the files it may touch, or how many, plus a stop rule ("report a short plan first" for anything beyond a small edit); the decided direction, if the issue left anything open — kittens don't make product decisions; whether it's model-facing (its own scoped eval, posted to the PR) or not; the expected PR title (`type(scope): title`); and "report the PR URL, summary, gate result, and any questions" — it never contacts the code owner itself, questions come to you.
- Rules not yet merged into the SOP go in the brief itself (a kitten reads the SOP from `main`), and test any command before telling a kitten to run it — a silently broken one leaves it looking compliant while it isn't.
- Before picking up an existing issue, PR or branch, call `list_other_tasks` to check no other task already has it.

## 4. When a task fans out to several issues

Fan-out is leverage for one workload shape, not a general speedup — the wrong shape costs more than working the issues one at a time. Default to one kitten at a time; fan out only when the issues are independent (disjoint code areas), verification is parallel (`make fix check` and/or scoped `make eval-remote`), and each is substantial next to its setup cost.

- **The meta ticket.** One per fan-out, labelled `meta`, body a checklist of child issues; own it end-to-end, checking items off as PRs merge — any "what's left?" answer is a live query against it, never memory.
- **Waves.** Group by disjoint files; hold shared-file issues (migrations included — one migration-creating issue per wave) for a later wave. Wait for a wave to clear the merge queue before launching the next.
- **Heartbeat.** While a kitten is queued on the per-provider eval lock (`make eval-remote` runs one eval at a time per provider; its log prints `eval: provider <name> is busy … waiting (<s>s of 3600s)`), check every 30–60 minutes that it's still live and its artifact is progressing — a dead waiter never resurrects itself. Size each kitten's `EVAL_WORKERS` so, across providers, the combined in-flight stays near 40.
- **Watch `mergeable`, not just open/merged** — a PR can go CONFLICTING when a sibling merges in the gap between its fetch and its push.
- **Fleet-end sweep.** Once every PR is terminal, inventory `git worktree list` against PR states, clean up every terminal one (§5), close the meta.
- **Bootstrapping onto one already running:** read the meta and its comments, `gh issue view`/`gh pr list` every child, `git worktree list` for live trees; a locked tree belongs to another live agent — relay, never force.
- **Stop every kitten at once if the code owner pauses work** — their worktrees keep the work for later.

## 5. Post-merge cleanup

Cleanup is yours, once a kitten's PR reaches a terminal state (merged or closed) — a kitten's own brief has it stop at its report and never clean up after itself. Remove its worktree and delete its local and remote branch (you may hand this mechanical step to a Haiku kitten). A locked worktree belongs to a live agent: leave it and relay the merge event instead of forcing it.

Still ask the code owner first: anything touching production (`make prod`/`up`/`kill`, the live DB), a whole-suite eval run, and anything outward outside the repo.
