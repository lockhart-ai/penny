# Agent Supervisor Runbook

How a Claude session **runs a fleet**: one supervisor owns a meta ticket, dispatches one subagent per child ticket, and shepherds the whole thing to done. This is the supervisor-side counterpart to `docs/agent-task-workflow.md` (the child's contract). The duty list lives in `CLAUDE.md` → *Agent Supervision*; this is the operating procedure.

The design: **GitHub is the durable state, not your context.** A fresh session must be able to pick up a half-finished fleet from the meta ticket + `git worktree list` alone. Record everything that matters (dispatches, PRs, decisions) on the meta as it happens.

## 0. Should this be a fleet at all? (decide before dispatching)

Fan-out is **not** a general speedup — it is leverage for a specific workload shape, and the wrong shape makes it *slower and far more expensive* than one session working the tickets in sequence. Decide deliberately; the default for a small or eval-gated batch is **one sequential session, not a fleet**. (A single session still works in a worktree per `CLAUDE.md` → Working Directory Discipline — that's isolation, not fan-out.)

The three failure modes to weigh — each observed to make a real fleet cost far more than the sequential alternative for no throughput gain:

- **Coordination can outweigh the work.** Dispatching, heartbeating, relaying, and shepherding all cost tokens that never touch code. When the changes are small, this fixed per-ticket overhead dwarfs the diff — and a fleet pays it once per ticket instead of amortizing it once across a whole session.
- **A fleet cannot outrun its slowest serial resource.** If every child must pass a single-tenant gate — the eval GPU — the "parallel" agents just queue behind it and sit idle. The parallelism is illusory at the bottleneck; adding agents only lengthens the line.
- **Cold starts and idle-polling dominate the token bill.** Each fresh subagent re-derives the world (reads the SOP, greps the tree, orients in git), and an agent blocked on a slow check burns turns re-reading its entire context just to wait. Many cold contexts cost dramatically more than one warm session that reuses its own.

**Fan out when ALL hold:** tickets are genuinely independent (disjoint code areas), verification is fast/local/parallel (`make fix check` only — **no GPU eval**), and each unit is *substantial* relative to its setup (a mechanical migration across many files, a broad audit, per-module refactors).

**Do NOT fan out when ANY hold — do it in one sequential session instead:** the units are small (a run of 10-line prompt/eval tweaks), verification is gated on the serial eval GPU, or the tickets touch shared files. A batch of tiny eval-gated PRs is the anti-pattern this gate exists to catch.

If it's a mixed bag, split it: sequence the small/eval-gated tickets in one session, and fan out only the independent `make fix check`-only cluster.

## 1. Bootstrap (fresh session, cold start or resume)

1. Find the meta: `GH_TOKEN=$(make token) gh issue list --label meta --state open`. Read it **and its comments** — decisions and dispatch records live there.
2. Reconcile reality: `gh issue view` every child ticket (state), `gh pr list --state all` for their PRs, `git worktree list` for live trees. Never trust a prior session's summary — query.
3. Inventory other sessions: locked worktrees (`git worktree list --porcelain`, lock reason names the pid) mean **another supervisor's live agents**. Don't grab another supervisor's meta or force-remove its locked trees without the user's say-so.
4. If any prior child's PR went terminal while unattended: relay the event (its agent may be resumable) or sweep it yourself (fleet-end sweep rules, CLAUDE.md).

## 2. The meta ticket

- One meta per fleet, labeled `meta`. Body carries a **task-list checklist** of child tickets (`- [ ] #NNNN — summary`) so GitHub renders progress and reconciliation is mechanical.
- Tier the checklist by actionability (directly actionable / needs-a-decision / design-first `blocked`+`investigation`) — children must not be dispatched onto design-first tickets.
- The supervisor **owns it end-to-end**: check items off as PRs merge, comment when waves launch (which tickets, which PRs), record mid-flight decisions. Any "what's left?" answer reconciles against this checklist via live queries — never from memory.
- At fleet end: close it with a final status comment, or explicitly state what remains open and why.

## 3. Dispatching a child

- **One ticket per agent**, in its own **isolated worktree** (Agent tool `isolation: "worktree"`).
- **Model: Opus for implementation subagents.** The supervisor session (Fable) reserves itself for orchestration, investigation, review, and design; ticket implementation runs on Opus (Agent tool `model: "opus"`).
- The child's prompt is short and defers to the SOP. Shape:
  1. "You own ticket #N end-to-end. **Read `docs/agent-task-workflow.md` FIRST and follow it end to end** — including §8 (shepherd to terminal state) and §9 (terminal cleanup)."
  2. How to read the ticket (`make token` from the primary checkout, `gh issue view N --comments`).
  3. **The decided direction** — if the ticket lists options, the supervisor (with the user) picks before dispatch; children don't make product decisions. Include known constraints/traps from your analysis.
  4. Scope boundary; whether it's model-facing (eval contract) or not (no eval).
  5. "Report the PR URL + summary + gate result."
- Rules **not yet merged** into the SOP must be stated in the prompt (children read the SOP from `main`). And **verify the mechanism before relaying a rule** — test the command you're telling them to run (`make -n` is cheap); a silently-broken mechanism makes every child non-compliant while believing otherwise.

## 4. Wave planning

- Group waves by **disjoint code areas**. Hold shared-file tickets (`CLAUDE.md`, `prompts.py`, migrations, shared test fixtures) for a later wave or do them inline — parallel edits there just manufacture rebases.
- **One migration-creating ticket per wave** (migration numbers collide; rebase-only renumber policy).
- Default to **waiting for a wave to clear the merge queue before launching the next** (limits PR pileup; later waves branch from the truth). Children flag merge-when-ready (§8), so the user's approvals are the only manual step — a batch of approvals cascades through the queue unattended. The shepherd loop makes modest overlap survivable, not free.
- Watch cross-fleet overlap: if another session's fleet is active, compare target files before dispatching.

## 5. While children run

**Hands off the prod stack.** Agent sessions (supervisor and children) run only one-off `docker compose run --rm` against the main project (`make token`, `make check` — fine); never compose *lifecycle* commands (`make up`/`make prod`/`make kill`/`docker compose down`) — the production stack belongs to the user. If a compose command you did run gets interrupted, **re-run it to completion**: a half-finished teardown mints orphaned container/network state that breaks the next `make prod` (the `up` targets now self-heal with a preceding `down --remove-orphans`, but don't rely on it).

Standing duties (details in `CLAUDE.md` → Agent Supervision): **heartbeat** every 30–60 min while anything waits on a serialized resource; **stall recovery** ("check your result artifacts FIRST, then relaunch only what's missing"); **resource arbitration** (full-suite evals need explicit user approval; GPU contention is yours to surface); **relay merge/close events** so children run §9; file children's out-of-scope findings as new tickets under the meta.

## 6. Fleet end

1. Every child PR terminal → **fleet-end sweep**: inventory `git worktree list` against PR states; remove terminal trees, delete local+remote branches; locked trees belong to live agents — relay, never force.
2. Update the meta checklist; close it (or report what remains and why).
3. Fold process lessons into this runbook / the SOP / CLAUDE.md — that's how this document got every rule it has.
