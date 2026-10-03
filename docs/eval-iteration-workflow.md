# The Eval Iteration Workflow

The repeatable loop for changing **model-facing behaviour**: how **one** objective goes from agreed input/output pairs → a seeded world → measured rounds on a live PR → merged. It is the eval-side sibling of the task-agent SOP — walk it top-to-bottom for one objective's iteration the way a task agent walks `docs/agent-task-workflow.md` for one ticket. Distilled from the state-transition suite arc (#1828–#1863).

The golden rule underneath all of it: **one beat at a time, pairs before code, every change measured, every run posted to the PR — the code owner rules on the cases at the PR review.**

**The four docs are one system.** Each owns exactly one thing:

- **`docs/eval-iteration-workflow.md`** (this doc) — the **loop**: scope → pairs → seeded world → run → report → diagnose → iterate → merge.
- **[`docs/prompt-writing-guide.md`](prompt-writing-guide.md)** — **how to write** the model-facing text a round changes (plain words, numbered call steps, the canonical call notation).
- **[`docs/eval-case-design.md`](eval-case-design.md)** — the **contract a case is written to**: end-state assertions in three categories, everything the model emits measured as variance, and the porting checklist. Required reading before a case is written or ported.
- **[`docs/agent-task-workflow.md`](agent-task-workflow.md)** — the **contract** the implementing task agent follows for each round's ticket → worktree → gate → PR → cleanup.

---

## 0. Roles (who decides what)

- **The code owner owns the cases and rules at the eval PR review.** Their inputs, behaviour sentences and claims, and whether a case lands, are his calls — not inferred from a plan he read. His gate is the review, not each run; a full-suite run is the one run that waits for his explicit go-ahead.
- **The supervisor holds the beat's scope** and owns the internals the round changes (prompts, machine, harness), per [`CLAUDE.md` → Git Workflow](../CLAUDE.md#git-workflow).
- **Task agents implement.** One ticket each, `docs/agent-task-workflow.md` as their contract. They run their own scoped evals and post every report to the PR. Most work dispatches this way; the supervisor may implement a round directly when it wants certainty or more context.

## 1. Scope one beat

- A **beat** is one edge, one micro-context, or one behaviour — never a family of them.
- One beat gets **one ticket and one PR**, and iterates on that PR until the code owner lands it.
- Findings about **later** beats are **recorded on their tickets, not chased**. Discovering the next problem is not permission to start it.

## 2. Design the input/output pairs — in chat, first

**The pairs ARE the tests.** Agree them in conversation before anything is written:

- **Variations on a theme.** One scenario structure, five-ish instances varying exactly the axes under test — cadence forms, expiry present/absent, one vs two parameters, subject domains. **Name what each variation stresses.**
- **The full register.** Each input carries everything the real user shape carries (a watch ask names a source, a store/compare intent, a cadence, a notify clause). A thin input proves less.
- **Audit the wording for double readings.** "Tell me the second something new shows up" cost **three rounds** before the ordinal reading was spotted. Use plain temporal words; if a phrase can be read two ways, the model eventually will.
- **Reference replies are review targets, never strings a claim matches.**
- The agreed pairs go into the ticket **verbatim** — the ticket body is the canonical contract, **amended in place** when a ruling changes it, never layered with update notes.

## 3. Encode with entrance-condition fidelity

The seeded world mirrors the **full exit state** of everything that came before, as production would have left it:

- **The whole conversation** — every turn of the preceding journeys, with Penny's replies **THREADED (`parent_id`)** exactly as the production reply path stamps them. An unthreaded reply is **invisible to the history builder**: nineteen of them once collapsed into one giant user turn.
- **The whole ledger** — promptlog rows for the preceding turns (real call shapes, under **`seeded-` run ids the harness excludes**), run stamps on entries / collections / mutations / skills, browse-results pages, transition rows with their run and message links, and the structural resets on the messages that settle them.
- **Time-shape** — recent turns are recent (banter after journeys), old work is out of the window. Order through the **real store APIs**; report what can't be controlled rather than faking it.
- **Fixtures are transcriptions of measured draws, never inventions.** Sample the clean modal draw; keep subject vocabulary intact ("daily special" is the dish's name); **re-sample whenever the producing prompt changes**.
- **Pages are richer than the task needs** — a real listing has an author, a blurb, older entries. A sparse page turns an over-asked extract into a false dead end.
- **A loud seed probe asserts the whole world**, so a bad world fails **in the seed, not after GPU time** — plus a plain `make check` test driving the composed seeder against a migrated DB, including the built conversation window's **actual alternation**.

## 4. Run

- **One scoped run per round:** `-k` the beat's case test functions only (`-k` matches pytest node ids, not case ids). **A full-suite run is a question to the code owner, never bundled** into the round.
- **`EVAL_LEVER` states the round's one hypothesis.** `EVAL_BASELINE` points at the prior round, so every change renders as an explicit flip. `N` covers the pool **exactly once** where pools rotate.
- **Verify before measuring** — PR head == worktree HEAD, and the run's process cwd **is** the worktree. A number describes the tree that produced it, not the PR you think it belongs to.
- **Never edit the worktree while a run executes from it.**

## 5. Report

- **Every run posts its verbatim report to the PR** (`make eval-report PR=<n>`) — the comment stream **is** the durable record of the iteration. What a ported case's report says: [`docs/eval-case-design.md`](eval-case-design.md); how it gets posted: [`docs/agent-task-workflow.md`](agent-task-workflow.md) §4.
- **The chat summary follows one shape, in this order:**
  1. **Overall** — claims passed of claims checked, per model, with the baseline beside them.
  2. **Per-case table** — each claim's passing count, each row naming the case's behaviour, plus run health and any variance feature that rose.
  3. **Failures as observed symptoms**, with sample counts.
  4. **Diagnoses last**, each tied to the symptoms it explains.
- **Claims are reported as passing counts (`14/15`), never as a mean.** A claim is true or false of each sample, so a count needs no gloss, and no count carries a floor ([`docs/eval-case-design.md`](eval-case-design.md) §1–§2).

## 6. Diagnose

- **Read the thinking and the ledger — never transcript greps.** Draws come from the **per-sample DBs**; the posted report is authoritative for the counts; use **explicit absolute run-dir paths**. A sample's `.db` and `.log` carry the number the report calls it — `sample 11` is `<case>-11.db` — so open the file the report names and no other.
- **Causes partition**: behavioral / pathology / harness. Then the behavioral ones partition again: **model choice · claim strictness · fixture defect · presented-state defect.**
- **The model's reasoning is ground truth.** When a draw is wrong, ask first **what the state failed to present**. The window collapse, the required-looking optional field, and the one-shot-verb description were all "model failures" that decoded as **correct reasoning over defective presentation**.
- **Verify the mechanism, not the inference** — dump the actual history window, parse the actual schema, count the actual draws. **Two diagnoses in this arc reversed on the raw read.**
- When the thinking shows a **coherent alternative reading of the contract**, the claim is asserting something the model may legitimately say another way: it shrinks to the smallest unique datum or moves to variance ([`docs/eval-case-design.md`](eval-case-design.md) §2), **never a per-case carve-out**. Changing a claim is the code owner's call at review.

## 7. Iterate on prompts

- **Short, simple, direct.** "Do not include any information about timing, scheduling, or notifications." beats every fact-then-instruction elaboration. **State schemas plainly**; add **permission clauses** where the model argues itself out of a rule; **name the boundary case**.
- **One lever per run.** The **second** patch to a prompt means the next change is a **wholesale rewrite**.
- **Choice-menu conditions:** each edge states only its **own** shape — no edge describes a sibling, no "but not when" carve-outs. Naming your **own** near-misses is fine.
- **Watched deletions:** a dropped guard is **recorded with the gate cases that would catch its absence**.
- **Expect the see-saw.** A tightening captures a neighbour; a loosening leaks one. **Two or three measured rounds to converge is normal**, and the guard cases run **in the same round as the fix**.
- **Prompt wording is an internal**, the supervisor's to write and approve ([`CLAUDE.md` → Git Workflow](../CLAUDE.md#git-workflow)). Text the code owner hands over lands verbatim, and **his register wins**.

## 8. Dispatch to task agents

- **One ticket, one scope, the SOP as contract.** Rounds land on the **inflight PR**; separate concerns get **separate tickets and PRs**. **Ask when unsure — never collapse "separate PR" into "inflight round" on your own reading.**
- **Code-owner-authored text lands VERBATIM.** The task agent verifies it **char-for-char by independent AST extraction after `make fix`**, sweeps **every carrier** (pins, docstrings, comments), and **reports judgement calls and out-of-scope findings instead of absorbing them**.
- **Task agents run their own scoped evals** and post every report to the PR (`docs/agent-task-workflow.md` §4). Most rounds dispatch; the supervisor may run a round itself when it wants certainty or more context.

## 9. Rerun upstream

- A change to **shared seeders, shared claims, shared prompts, or anything the model reads** = a **non-regression rerun of every touched suite, baselined**.
- When a **producing** prompt changes (the framer), the suites whose fixtures transcribe its draws need, in order: a **fresh sampling run** → the **fixture re-sample** → the **composed rerun**.
- **The final state before merge is one composed run of the beat on the finished tree.** A PR that adds or changes eval cases merges only with **both roster models run and posted at its final head**; a rebase that leaves the tree byte-identical to the measured head keeps the runs valid, and the PR says so in a comment (`docs/agent-task-workflow.md` §8).

## 10. The loop itself

**Change → scoped run → report on the PR → read → next round → PR review → ruling.**

- **The code owner's checkpoint is the eval PR review, not each run.** Scoped runs are the implementer's to run as the round needs; a full-suite run waits for his explicit go-ahead.
- **One beat to its terminal state.** Whether a case lands is his call.
- **Merged means the task agent cleans up and the next beat opens.**

---

## Invariants (true at every step)

1. **One beat per iteration; later-beat findings are recorded, not chased.**
2. **The pairs are agreed in chat and land in the ticket verbatim** — amended in place, never layered.
3. **The seeded world is the full exit state of what came before** — threaded replies, the whole ledger under excluded `seeded-` run ids, real time-shape, fixtures transcribed from measured draws.
4. **Every run is scoped to the beat's case tests and declares one lever**, baselined against the prior round; a full-suite run is a question, never a bundle.
5. **Verify head == worktree HEAD before measuring** — a number describes the tree that produced it.
6. **Every run posts its verbatim report to the PR**, and the chat summary runs overall → per-case → symptoms → diagnoses, in passing counts.
7. **Diagnose from the thinking and the ledger, never greps**; the model's reasoning is ground truth and presented state is the first suspect.
8. **One lever per run; the second patch means a wholesale rewrite**; the code owner's register wins on text he hands over.
9. **Code-owner-authored text lands verbatim, AST-verified**, and task agents run their own scoped evals.
10. **Anything shared that changed gets a baselined non-regression rerun**, ending in one composed run of the beat on the finished tree; an eval-case PR merges with both roster models posted at its final head.
11. **The code owner's checkpoint is the eval PR review** — not each run; the full suite still waits for his go-ahead.
