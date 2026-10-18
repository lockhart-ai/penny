# The conversation-machine-and-skills era

A whole-tree snapshot of `penny/CLAUDE.md` and `docs/reference/*.md` as they stood at
`80d1a641` — the commit immediately before #2537 retired them. Taken as a unit rather
than restored file-by-file, because the era is a single coherent design: a model-filled
conversation state machine and a taught-skill framework built on top of it, read together.

## What the era was

A conversation lived in a `ConversationState` (idle · elicit · learn · request · apply),
walked by an edge table the model filled in rather than chose from directly. On top of
that machine, a user-taught routine was framed, bound, and distilled into a named
**skill** — its own substrate, store, and eval surface — so a demonstrated sequence of
tool calls could be replayed later without the user re-teaching it each time.

Built mainly across:
- **#1590 / #1599** (2026-07-13) — the skills substrate: a dedicated table, skill
  creation by reference, hole inference, steps-to-text rendering.
- **#1706** (2026-07-22) — the conversation state machine itself: the edge table,
  snapshot, and classifier micro-context.
- a run of framer/binder/namer and skill-optimisation work through August and September
  2026 (see `docs/archive/README.md`'s timeline for the surrounding eras).

## What retired it

**#2537** (2026-10-18), "naive jobs — no conversation machine or skills; a job runs the
prompt it was given." The machine and the skill framework were both removed outright: a
job is now a plain collection with a free-text prompt, a schedule, an expiry, and a
notify flag, run as written with no rendering seam in between.

## Where the live docs are now

`penny/CLAUDE.md` and `docs/reference/*.md` describe the naive job runtime. The
conversation-machine reference itself (#2541) lives in this snapshot's own
`docs/reference/conversation-machine.md`, beside its siblings here, for the full
picture of what the era covered.
