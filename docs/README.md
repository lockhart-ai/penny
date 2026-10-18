# Docs — a map of every doc by kind

Every doc under `docs/` (outside `docs/archive/`, which has its own index), grouped by
kind, with one line on when to read it. Start here before opening any one of them.

## North star

- [`principles.md`](principles.md) — Penny's core design principles; read first whenever a
  question about architecting a mechanism or a test comes up. Wins over every other doc
  here when something disagrees.

## Reference (how a part of Penny works now)

- [`reference/agents.md`](reference/agents.md) — the agent loop, ChatAgent, the shared LLM
  clients, the self-state header; read before changing `penny/agents/` or `preflight.py`.
- [`reference/channels.md`](reference/channels.md) — the channel abstraction and each
  concrete channel (Signal, Discord, the browser extension); read before changing a
  channel or the outgoing-send path.
- [`reference/database.md`](reference/database.md) — the data model and the store classes;
  read before changing a table, a store's read/write shape, or the clock.
- [`reference/eval-harness.md`](reference/eval-harness.md) — the live-model eval harness's
  running, reporting and posting mechanics; read before changing how a run is scored,
  reported, or posted.
- [`reference/scheduler-and-collector.md`](reference/scheduler-and-collector.md) — the
  scheduler, the Collector dispatcher, and collector-cycle mechanics; read before changing
  scheduling, collector cadence, or a cycle's end conditions.
- [`reference/tools.md`](reference/tools.md) — the Tool ABC, BrowseTool, every
  micro-context customer, and the memory tool surface; read before changing a tool's
  arguments, a micro-context's output contract, or the write-gate.
- [`reference/validation.md`](reference/validation.md) — the model-I/O validation taxonomy
  and the content-validity predicates it's built from; read before changing a response
  validator or a degeneracy/half-formed-send rule.
- [`model-io-validation.md`](model-io-validation.md) — the formal structure behind the
  informal checks on a model's JSON response (use, nudge, repair, reject, stop); read
  before changing that decision logic, as the counterpart to the static tool-argument
  validators.
- [`browser-extension-architecture.md`](browser-extension-architecture.md) — the browser
  extension's vision and architecture; read before changing `browser/` or the browser
  channel's design.

## Process (how work is done)

- [`agent-task-workflow.md`](agent-task-workflow.md) — the kitten SOP: one GitHub issue
  from a brief to a reviewed, green draft PR; read before starting any kitten work.
- [`agent-supervisor-runbook.md`](agent-supervisor-runbook.md) — the coordinator's runbook:
  diagnose, file the issue, brief a kitten, watch it, review, merge, clean up; read before
  coordinating a Glade task.
- [`pr-review-guide.md`](pr-review-guide.md) — the canonical, exhaustive PR review
  checklist for the main repo; read before reviewing or self-checking a PR.
- [`penny-client-agent-task-workflow.md`](penny-client-agent-task-workflow.md) — the kitten
  SOP for PennyClient (iOS) tasks specifically; read before taking a `penny-client/` issue.
- [`penny-client-pr-review-guide.md`](penny-client-pr-review-guide.md) — the focused
  checklist for reviewing Swift/iOS changes in `penny-client/`; read alongside the
  repo-wide review guide when reviewing a client PR.
- [`prompt-writing-guide.md`](prompt-writing-guide.md) — how to write the model-facing
  prompts that drive Penny (extraction prompts, system prompts, tool descriptions); read
  before authoring or revising a prompt.
- [`eval-working-guide.md`](eval-working-guide.md) — the working method for planning a
  case or a change, running it, reading what came back, and deciding what to fix; read
  before iterating on eval cases.
- [`eval-case-design.md`](eval-case-design.md) — the contract for what one eval case is
  allowed to claim; read before writing or porting a case.
- [`eval-iteration-workflow.md`](eval-iteration-workflow.md) — the repeatable loop for
  changing model-facing behaviour, one beat at a time, pairs before code, every change
  measured; read before starting a behaviour-change iteration.

## History (records kept as they grew)

- [`eval-suite-history.md`](eval-suite-history.md) — a case-by-case, file-by-file record of
  how the live-model eval suite's cases and reporting mechanics came to be; read on demand,
  not as onboarding.
- [`migration-log.md`](migration-log.md) — a migration-by-migration record of
  `penny/database/migrations/`; read on demand, not as onboarding.

## Archive

- [`archive/README.md`](archive/README.md) — retired designs as a dated timeline;
  documentation that stops describing Penny moves here, never to the bin.

## Generated, not committed

- `eval-catalogue.md` — every eval case by area, and which cases cover each machine edge;
  generated on demand by `make catalogue` (gitignored, never committed).
