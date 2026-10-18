# Archive — a dated history of Penny's designs

Documentation that stops describing Penny moves here, never to the bin: a design that
was tried and later replaced, a plan whose feature shipped and whose write-up is now
history rather than a living doc, a format a later ruling superseded. Deleting a doc
erases the record of what was tried; archiving it keeps that record so the project's
history can be told later. Each entry below is where a design came from, what it tried,
and — where it has one — the PR that retired it and what took its place. Oldest first.

## 2026-02-05 — Feature TODO list

- [`TODO.md`](TODO.md) — an early features/enhancements wishlist (new agent types,
  digests, fact-checking) sketched before GitHub issues became the project's backlog.
  Never described Penny as built; moved here (#2546) once it had long stopped being
  read as a living list. The GitHub issue tracker is the live backlog.

## 2026-02-08 — penny-team: an autonomous dev team

- [`penny-team/CLAUDE.md`](penny-team/CLAUDE.md), [`penny-team/README.md`](penny-team/README.md),
  and the [`architect`](penny-team/architect/CLAUDE.md), [`monitor`](penny-team/monitor/CLAUDE.md),
  [`product-manager`](penny-team/product-manager/CLAUDE.md), and [`worker`](penny-team/worker/CLAUDE.md)
  role briefs — a scheduled pm/architect/worker/monitor/quality team of Claude CLI agents
  that processed GitHub issues on its own. Renamed from `agents/` in #82 (2026-02-08);
  restructured into its current per-role layout in #84 (2026-02-08), which dropped the
  original `penny-team/README.md` as stale (restored here anyway, for the record).
  Removed whole in **#1598** (2026-07-12): it had gone unused and its per-worktree
  Docker images kept filling disk. Replaced by Glade-task kittens — see
  `docs/agent-task-workflow.md`.

## 2026-02-17 — Entity/fact knowledge system

- [`knowledge-system-plan.md`](knowledge-system-plan.md),
  [`knowledge-system-flows.md`](knowledge-system-flows.md), and
  [`events-system-plan.md`](events-system-plan.md) — a design for extracting entities
  and facts from conversation into a structured knowledge store, with sequence-diagram
  flows for how it would be read back. Written for #294 (2026-02-17). Removed in
  **#653** (2026-03-06): the reasoning-pipeline cost didn't earn its keep. Replaced by
  the preference/memory system described in `docs/reference/database.md`.
- [`agent-memory-patterns.md`](agent-memory-patterns.md) — written alongside the
  knowledge-system docs (#294) as a practical reference on memory-architecture patterns
  (working memory, episodic memory, semantic consolidation). Not a retired design: it
  outlived the knowledge system and is kept here as a **standing reference**, not a
  historical record of something Penny once did.

## 2026-02-26 — Model benchmarking notes

- [`benchmarking-qwen35-vs-gpt-oss.md`](benchmarking-qwen35-vs-gpt-oss.md) (#469) and
  [`benchmarking-embedding-models.md`](benchmarking-embedding-models.md) (#470) — a
  point-in-time comparison of qwen3.5:35b against gpt-oss:20b, and of embedding models
  (settling on `embeddinggemma`). Moved here (#2546): the live model choice has moved
  on since, and these are a snapshot of that moment's measurements rather than current
  guidance.

## 2026-03-28 — Browser & multi-channel architecture plans

- [`channel-manager-plan.md`](channel-manager-plan.md) and
  [`browser-tools-plan.md`](browser-tools-plan.md) — the plan for unifying Signal,
  Discord, and the browser extension into one conversation history with device
  routing, and for the extension's tool-execution flow (permission model, domain
  allowlist, WebSocket protocol). Written for #896 (2026-03-28); both shipped. Filed
  here as finished plans in **#2340** (2026-10-14). The live shape is
  `docs/reference/channels.md` and the `browser/` extension itself.

## 2026-04-20 — Task framework & memory implementation plans

- [`task-framework-plan.md`](task-framework-plan.md) and
  [`memory-implementation-plan.md`](memory-implementation-plan.md) — a unified data
  layer and task framework to move Penny's behaviour from hardcoded Python into
  user-authored tasks and memories, and the twelve-stage rollout for the memory schema
  behind it. Written for #980 (2026-04-20); both shipped. Filed here as finished plans
  in **#2340** (2026-10-14). The live shape is `docs/reference/database.md` and the
  collection/log store it describes.

## 2026-06-14 — Prompt-validation harness, replaced by the self-improvement loop

- [`prompt-validation-README.md`](prompt-validation-README.md) — the out-of-band
  harness (`scripts/prompt_validation/`) that read prompts via AST, mirrored enums, and
  hand-built tools to validate them without running a real agent — and so drifted from
  real behaviour. Removed in **#1251** (2026-06-14).
- [`self-improvement-loop.md`](self-improvement-loop.md) — written the same day (#1251)
  as the plan for what replaced it: eval isolation, intent-as-spec, and the
  log-to-test-to-fix loop. Shipped as the live-model eval suite; filed here as a
  finished plan in **#2340** (2026-10-14). The live shape is `docs/eval-case-design.md`
  and `penny/penny/tests/eval/`.

## 2026-06-30 — Collector run-health reasoning plan

- [`collector-run-health-reasoning-plan.md`](collector-run-health-reasoning-plan.md) —
  structural signals and validation discipline so Penny has observable facts about a
  collector run's outcome, and reviews its own quality rather than asserting success.
  Written for #1292 (2026-06-30); shipped. Filed here as a finished plan in **#2340**
  (2026-10-14).

## 2026-07-03 — Prompt & recall efficiency plan

- [`prompt-and-recall-efficiency.md`](prompt-and-recall-efficiency.md) — a static/dynamic
  split of Penny's prompts to cut per-turn latency and context-window cost. Written for
  #1302 (2026-07-03); shipped. Filed here as a finished plan in **#2340** (2026-10-14).

## 2026-07-06 — Tool narration plan

- [`tool-narration-plan.md`](tool-narration-plan.md) — a self-narrating tool-result
  contract: tagged first-person framing so a tool's result reads as part of the agent's
  own voice. Written for #1479/#1489 (2026-07-06); shipped. Filed here as a finished
  plan in **#2340** (2026-10-14).

## 2026-07-13 — Conversation machine & skills

- [`conversation-machine.md`](2026-10-conversation-machine-and-skills/docs/reference/conversation-machine.md) — the conversation state
  machine's reference (idle · elicit · learn · request · apply): the model filled
  states from an edge table rather than choosing a path directly, and a demonstrated
  tool sequence was framed, bound, and named as a replayable **skill**. The skills
  substrate shipped in #1590/#1599 (2026-07-13); the machine itself in #1706
  (2026-07-22). Retired whole in **#2537** (2026-10-18): "naive jobs — no conversation
  machine or skills; a job runs the prompt it was given." This reference moved to the
  archive in #2541 rather than being deleted.
- [`2026-10-conversation-machine-and-skills/`](2026-10-conversation-machine-and-skills/)
  — a whole-tree snapshot of `penny/CLAUDE.md` and the rest of `docs/reference/` as
  they stood immediately before #2537, so the era reads as a coherent unit rather than
  scattered single-file diffs. Also from **#2537**; see that directory's own README for
  the era's detail. Replaced by the current `penny/CLAUDE.md` and `docs/reference/*.md`,
  which describe the naive job runtime.

## 2026-08-29 — Eval report format, superseded by the case-design doc

- [`eval-report-format.md`](eval-report-format.md) — the original posting and grading
  grammar for eval reports, predating the ratified case design. Superseded in **#2012**
  (2026-08-29) once that design moved to `docs/eval-case-design.md`, where a porting
  agent meets it first.
