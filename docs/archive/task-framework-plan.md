# Memory + Task Framework Plan

## Overview

Transform Penny's data and behavior layers in two phases:

1. **Phase 1 — Memory.** Unify all persistent user-meaningful data into a single primitive: memories. Every existing specialized store (messages, preferences, thoughts, knowledge) becomes a memory. Existing agents migrate to broker all their reads and writes through a narrow, uniform tool surface. All data-layer interactions move into model-space tool calls.

2. **Phase 2 — Task Framework.** Once the data layer is stable and agents already operate via collection tool calls, lift each hardcoded agent into a user-authored *task* — a named, scheduled, editable system prompt + tools + success criteria. Behavior joins data in being reshapable through conversation rather than through code.

Phase 1 is a shippable refactor on its own. It unlocks user-created memories ("track my weight," "keep a recipe log") as a real capability immediately, independent of Phase 2. Phase 2 is designed after Phase 1 lands — its shape will be informed by what we actually learn from running migrated agents on the uniform data layer.

The design aims for the flexibility of fully autonomous agents while staying within safe primitives: typed memories instead of filesystem, domain-gated tools instead of raw shell, inner prompt edits instead of root prompt edits.

---

## Motivation

### What's wrong with the current architecture

Penny today encodes one specific theory of what a personal assistant should do — extract preferences, generate thoughts, summarize knowledge, send notifications — and bakes that theory into Python. Every store (`preference`, `thought`, `knowledge`, `entity`) has its own schema, its own dedup logic, its own retrieval patterns, its own mutation rules. Each agent knows where to write by calling table-specific methods; each retrieval site hand-codes its own similarity-sort-and-inject pipeline. Adding a new behavior means writing a new table, a new memory, a new retrieval, and a new orchestrator.

A different user (e.g., small-business flows — email triage, schedule comparison) gets our theory whether it fits or not. Every new behavior costs a PR, CI, review, deploy cycle.

### What a unified data layer unlocks

If all user-meaningful data lives in memories with uniform shape, uniform retrieval, and uniform tool access:

- Users construct arbitrary data stores by describing them to Penny ("make me a weight log")
- All agents use one access pattern — no more store-per-table boilerplate
- Ambient retrieval extends to new memories automatically (just embed the content)
- Failed experiments are cheap (drop the memory, not a migration)
- The door is open to Phase 2, where *behaviors* also become user-authored

### Why now

We've tuned the existing pipelines to a known-good baseline. Parity against them is the Phase 1 success criterion — concrete and testable. And we've validated the tool surface with a dry-run harness (see *Validation Status* below) before touching any production code, so the design is grounded in real model behavior, not speculation.

---

## Design Philosophy

### Python-space over model-space

The model does as little reasoning about mechanics as possible. Cursors, dedup, authorship, timestamps, rate limiting, and routing are all Python concerns. The model reasons about *content* — what to extract, what to summarize, what to write — not about bookkeeping. Every time we can move a concern from model-space to Python-space, we do.

### Determinism where it matters, flexibility where it helps

- **Triggers are deterministic** (scheduled or event-driven, decided by Python).
- **Data operations are deterministic** (tool calls with narrow, typed args).
- **Content generation is model-driven** — the one place we actually want the model to be creative.

### State is collection membership

There are no mutable fields on entries. Content is immutable. State transitions are expressed by moving an entry between collections: `likes` ↔ `dislikes`, `unnotified-thoughts` → `notified-thoughts`. This makes the state machine visible in the memory graph, eliminates per-entry state management, and preserves embeddings across transitions (moves don't re-embed).

### Everything is a tool call

Agents broker every data interaction through tool calls. There is no structured-output parsing, no internal Python logic that reads or writes on the model's behalf. The model calls `collection_read_*`, `collection_write`, `collection_move`, or `done`; Python implements those calls; the agent's entire data footprint is visible in the tool-call log.

### Narrow, purpose-fit tools

One tool per operation, unambiguous from its name. `collection_read_random`, `collection_read_latest`, `collection_read_next` are separate tools rather than one `collection_read(shape=...)` tool with a mode parameter. Small local models handle narrow tools with tight schemas dramatically better than modal ones. Tool-list content cost is negligible.

### Current pipelines as benchmark

Done = new collection-backed versions of preference / thought / notification / knowledge pipelines produce comparable output to the current hardcoded pipelines. No feature regression, but also no preservation of incidental state-management complexity — those are data-layer consequences of the migration (see *Data-layer consequences* below).

---

## Phase 1: Data Layer

### Two persistent data shapes: collections and logs

User-meaningful persistent data in Penny is one of two shapes:

- **Collection** — a named, keyed set of entries. Unique by key (with fuzzy dedup on near-matches). Accessed by key, similarity, random, etc. Examples: preferences, thoughts, knowledge, recipes, trip days, todos.
- **Log** — a named, time-ordered stream of entries. No keys. Append-only. Accessed by time range, recency, or similarity over content. Examples: messages, weight measurements, browse fetches, diet entries.

The two shapes correspond to genuinely different data: *things* vs *events*. Collapsing them into one primitive with a `unique` flag (earlier iteration) conflated the two — separating them makes each type simpler, with narrower tool surfaces and unambiguous semantics.

Both types share:
- **name** — stable identifier
- **description** — human-readable purpose; load-bearing (chat agent uses it to pick which memory to write to; gets injected into agent system prompts)
- **recall** — how entries appear in chat context: `off` | `recent` | `relevant` | `all`
  - `off` — never surfaced in chat
  - `recent` — latest entries by created_at
  - `relevant` — top-K by embedding similarity to current message (with a similarity floor — see *Ambient injection*)
  - `all` — every entry, unconditionally, every turn. For small always-on memories like user-defined rules, guidelines, or standing instructions ("never do X", "always respond in this style"). Total injected volume scales with entry count, so `all` is only appropriate for memories that stay small.
- **archived** — boolean, default `false`. When `true`, excluded from ambient recall entirely. Data preserved and still directly readable.

**`recall` and type are orthogonal.** Defaults are type-driven (collection → `relevant`, log → `recent`) but any combination is valid. A log with `recall: relevant` surfaces historically matching events by topic (e.g., `user-messages` matching current conversation). A collection with `recall: recent` surfaces whichever entries were added most recently regardless of topic (e.g., "currently-reading" books).

### How Penny picks the shape

The two questions Penny answers when creating a persistent memory for the user:

1. **Type** — "Is this a set of distinct things, or a stream of events?"
2. **Recall** — "Should this surface in chat, and by topic or by recency?"

Examples:

| User request | name | type | recall | Why |
|---|---|---|---|---|
| "keep my recipes" | `recipes` | collection | relevant | Distinct things; surface matching ones when cooking comes up |
| "track my weight" | `weight-log` | log | recent | Each weighing is a distinct event; "how's my weight?" wants latest |
| "remember friends' birthdays" | `birthdays` | collection | relevant | Distinct facts; surface relevant ones when planning |
| "daily journal" | `journal` | log | recent | Each entry is a moment; check-ins want latest |
| "private notes, don't bring up" | `private-notes` | collection | off | Distinct items, explicitly hidden |
| "quiet coffee log — don't mention" | `coffee-log` | log | off | Event stream, explicitly hidden |

### Entry shape

**Collection entry:**
- **key** (text, embedded) — short topic/identifier phrase (e.g., `cilantro`, `cyberpunk novel`, `day1`)
- **content** (text, embedded) — the text of the entry

**Log entry:**
- **content** (text, embedded) — the text of the event

System-maintained on both, not author-visible:
- **author** — `user | chat | task:<name> | system` — assigned by the hypervisor based on the calling context.
- **created_at** — timestamp.
- **embeddings** — content is always embedded. For collection entries, the key is also embedded separately so retrieval can match against either phrasing (a user's query like "cyberpunk novels" hits the key; "that book with the great worldbuilding" hits the content).

The writer only ever produces the content (+ key for collections). Everything else is system-tracked.

### Why content is immutable

Content is what the embedding was computed from; changing it would invalidate the embedding and any retrieval that used it. The mutable-seeming needs of the old schema are handled without mutation:

- **Categorical state (collections only)** — a collection split plus `collection_move`. Valence becomes `likes` / `dislikes`. Sent status becomes `unnotified-thoughts` / `notified-thoughts`.
- **Observed history** — that's what logs are for. Each cooking of a recipe is a new entry in a `recipe-cookings` log; "how many times cooked" is a count.
- **Genuine edits to a collection entry** — `collection_update(collection, key, new_content)` (internally delete + write).

### Dedup (collections only)

Logs are append-only; duplicate content is legitimate (two weighings of the same number, two identical messages). No dedup.

Collection writes are dedup-checked. The check combines key-similarity and content-similarity signals, with thresholds tuned empirically:
- key match above a threshold
- content match above a threshold
- (key + content) combined match above a threshold

If any of those hit, the write returns `"duplicate found, write failed"`. Otherwise `"ok"`. The model never configures dedup; it's uniform behavior across all collections.

Cross-collection dedup (e.g., "don't re-say a thought that was already notified") is handled via the explicit `exists(collections=[...], key, content)` tool that the model calls before writing.

### Tool surface

All implemented as narrow Python wrappers over the underlying memories.

**Discovery (both types):**
- `list_memories()` — names, descriptions, and types (collection | log) of everything that exists
- `describe_memory(name)` — full metadata for one memory

**Collection reads:**
- `collection_get(collection, key)` — direct key lookup
- `collection_read_latest(collection)` — most recent entries, Python-capped
- `collection_read_random(collection, k)` — `k` random entries
- `collection_read_similar(collection, anchor, k)` — top-`k` by embedding distance (scored against both key and content)
- `collection_read_all(collection)` — every entry
- `collection_keys(collection)` — just the keys, no content (cheap discovery)

**Collection writes:**
- `collection_create(name, description, recall, archived=false)` — create a new collection
- `collection_write(collection, entries=[{key, content}, ...])` — batch write; returns per-entry `"ok"` / `"duplicate found, write failed"`
- `collection_update(collection, key, content)` — replace the content of an existing entry
- `collection_move(key, from_collection, to_collection)` — state transition; preserves content and embeddings
- `collection_delete(collection, key)` — remove
- `collection_archive(collection)` / `collection_unarchive(collection)` — flip the `archived` flag

**Log reads:**
- `log_read_latest(log, k)` — last `k` entries
- `log_read_recent(log, window)` — entries within a time window
- `log_read_since(log, timestamp)` — entries after a cursor; used for incremental processing (Python-managed cursor per agent, advances on successful run completion)
- `log_read_similar(log, anchor, k)` — top-`k` by embedding distance to anchor text
- `log_read_all(log)` — everything (for small logs only)

**Log writes:**
- `log_create(name, description, recall, archived=false)` — create a new log
- `log_append(log, entries=[{content}, ...])` — batch append; always succeeds (no dedup)
- `log_archive(log)` / `log_unarchive(log)` — flip the `archived` flag

Logs have no move, no update, no delete — they're append-only.

**Introspection:**
- `exists(collections=[...], key, content)` — boolean; true if any listed collection has a similar entry. (Logs don't participate — they're event streams, duplicate events are legitimate.)

**Lifecycle:**
- `done()` — explicit run-end signal; no args

**External (unchanged from current):**
- `browse(url)`, `search(query)` — web access via browser extension
- `send_message(content)` — rate-limited outbound to user

### Side-effect writes

Some tools write to memories as part of their behavior, without the model orchestrating the write:

- `send_message(content)` → appends to the `penny-messages` log (author=calling-agent identity)
- `browse(url)` → appends to the `browse-results` log with content=page text. Same URL re-fetched later is a new log entry; page content can change over time, both fetches are preserved.
- Channel adapter ingress → appends to the `user-messages` log

The hypervisor handles persistence implied by the tool's contract. The agent doesn't think about it.

#### Default: tools that produce content get a log

The rule isn't "log tools where we have a planned consumer" — it's "log tools where a consumer could plausibly be user-authored later." Once Phase 2 lands and users can create their own agentic loops through conversation, every tool output is a potential input stream. We can't predict which ones the user will compose over, so the default is to log.

Expected default-on tools (present and future):
- `browse` → `browse-results`
- `send_message` → `penny-messages`
- channel ingress → `user-messages`
- `search` → `search-results` (user might want to analyze search patterns or compose loops over queries)
- `/draw` → `images-generated` (with media-token references)
- future email tools → `email-reads`
- future calendar/Zoho tools → their own logs

Explicit exceptions are for tools that truly produce nothing worth retaining — which, in practice, turns out to be a short list. Most tools produce something someone eventually wants to reason about.

The knowledge extractor, in this framing, is just the first-shipped specific instance of a general pattern: a task that reads a log (`browse-results`) and writes to a collection (`knowledge`). In Phase 2, the user can create new task-reads-log-writes-collection loops over any logged tool output — no code, just conversation.

### Log-since cursor mechanics

For incremental processing patterns (preference extractor reading new user messages, knowledge extractor processing new browse results), `log_read_since(log, timestamp)` is the primary read. Python manages per-`(agent, log)` cursors in an `agent_cursor` table:

1. Agent's system prompt resolution substitutes a `since_last_read` token (or similar) with the current committed cursor before the run.
2. Agent calls `log_read_since(log, cursor)`. Returns up to N entries ordered ascending. N default 10; configurable per agent.
3. Record a pending cursor advance in memory (max created_at of returned batch).

On successful run completion: commit pending advances. On failure: discard them. At-least-once processing — if an agent crashes after writing but before committing, collection dedup-on-write catches any redundant reprocessing of derived outputs.

Within a run, subsequent reads use `max(db_cursor, in_memory_pending)` so pagination works within-run as well as across-runs.

### Ambient injection (outer chat context only)

When the chat agent assembles context for a user turn, the hypervisor queries each memory (collection OR log) that is not archived and whose `recall` is not `off`, then unions the results (tagged by source) into the system prompt:

- **`recall: recent`** — latest N entries by `created_at` (K a system-wide default, e.g. 5)
- **`recall: relevant`** — top-K by embedding similarity to the current user message, with a similarity floor. Entries below the floor don't surface. A memory that has no entries above the floor for this turn contributes nothing — its slot is effectively skipped. This keeps ambient context bounded even as the user accumulates many relevant-mode memories over time; stale or off-topic memories naturally fall out without anyone having to archive them.
- **`recall: all`** — every entry from the memory, injected unconditionally. For small always-on memories (rules, guidelines, standing instructions). No similarity check, no cap — everything is considered perpetually relevant to every interaction.

A single memory uses one mode, not both. Different memories can use different modes in the same chat turn — the chat context is the union of "recent conversation" (messages log, recall=recent) plus "relevant historical user statements" (user-messages log, recall=relevant) plus "relevant preferences" (likes/dislikes collections, recall=relevant) plus whatever else.

Between the similarity floor (automatic) and the `archived` flag (explicit), ambient context volume stays controlled as the memory set grows.

Recall results are rendered as a structured block appended to the system prompt, something like:

```
## Recall (entries relevant to this message)

Collection: london-trip (Trip to London for work week)
  - [tuesday] WeWork Shoreditch High Street – coffee: Allpress…
  - [wednesday] WeWork Soho / Aldwych – coffee: Monmouth…

Collection: likes (things the user likes)
  - [cyberpunk novels with strong worldbuilding] …
```

This is how the chat agent stays continuous across sessions without passing conversation history. When the user says something like "Tuesday's WeWork" in a fresh session, the trip collection's Tuesday entry is already in the agent's context by the time it reads the user's message — no separate read call needed, no guessing, no clarifying question. **Ambient recall isn't a context-enrichment nice-to-have; it's the mechanism that makes cross-session chat work at all.**

Background agent contexts do **not** get ambient injection. They see only their declared system prompt, their explicit tool-call returns, and the memory list. This preserves the fixation-prevention property — a thinking loop isn't flooded with its own prior output.

### Agent system prompt structure

Every migrated agent gets its system prompt assembled by the hypervisor as:

```
{agent-specific job prompt}

## Available memories

- user-messages: messages from the user
- penny-messages: your past responses
- likes: things the user likes
- dislikes: things the user dislikes
- unnotified-thoughts: thoughts generated but not yet shared
- notified-thoughts: thoughts already shared
- knowledge: factual summaries from browsing
- browse-results: pages that have been fetched (source for knowledge extraction)
...

## Tools

{rendered tool descriptions for the tools granted to this agent}
```

The memory list is just every memory that exists — assembled at prompt-build time from the memory table. New memories show up automatically in every agent's next invocation.

Every agent gets the full tool surface. The agent's system prompt describes its job; the model decides which tools apply to that job. We don't pre-subset the menu because that's a judgment call the model is capable of making from the prompt, and hardcoding per-agent subsets reintroduces the "different agents have different capabilities" complexity the platform is meant to eliminate.

### Access model

Any agent can read or write any memory. Penny is single-user; there's no adversarial principal to guard against. Safety bounds live at the **tool** layer:
- `send_message` is rate-limited
- Tools only expose user-meaningful operations — filesystem, shell, and system tables are not reachable from any tool
- A misrouted write is a prompt bug visible in the run log, fixable in conversation

### Media store

Images and other binary content don't live inside `content` fields (text only) or as collection entries. They go in a separate system table — the **media store** — and are referenced from text content via tokens.

- **Ingress**: channel adapters receiving images (Signal attachments, browser extension images, etc.), the `browse` tool returning embedded page images, the `/draw` tool generating images — all write bytes into the media store and get back an id.
- **Token**: a compact marker like `<media:42>` embedded in whatever text content would reference the image. The token is just a string to the model and to all downstream text operations — summarizers, embedders, recall injectors, log appends — they just carry it along.
- **Egress**: channel adapters on outbound scan message text for media tokens, fetch bytes from the media store, and attach them to the outgoing message in the channel's native format (Signal attachment, Discord embed, browser image, etc.).

Schema shape:
```
media
  id            INTEGER PRIMARY KEY
  mime_type     TEXT
  bytes         BLOB       (or external path for large media)
  source_url    TEXT       (optional)
  created_at    DATETIME
```

The media store is infrastructure, not a user-facing memory. Agents don't call `media_create` or `media_read`; they receive images via tool results that already contain tokens and emit content that still contains tokens. The swap happens at the channel layer.

Token format needs to be chosen deliberately to survive JSON serialization cleanly and not collide with normal text. Leading candidates: `<media:42>`, `[[media:42]]`, `{{media:42}}` — settle during implementation.

### Orchestration: handling model fumbles

Small local models reliably produce malformed or incomplete tool-calling sequences, especially on multi-step operations. The tool-dispatch orchestration must handle three failure modes without crashing the run:

- **Malformed JSON in tool call arguments** — catch the parse error at the API boundary, append a corrective message ("your previous tool call had malformed JSON; retry with strictly-valid JSON"), and retry the same turn.
- **Empty response with no tool calls** — when the model returns neither content nor tool calls, that's a signal it dropped the thread. Append a nudge ("you returned an empty response; complete the user's request") and retry.
- **Tool call with missing or invalid arguments** — dispatch must wrap each tool call in try/except, and return any error as a string result to the model (e.g., `"error: bad arguments for collection_update: missing 'content'"`). The model then sees the error in the tool result and can retry with corrected args.

All three patterns validated in the dry-run harness against gpt-oss:20b. They turn otherwise-fatal model errors into recoverable loop iterations. This is platform-level behavior — the chat agent, background agents, and future task runners all need the same retry discipline.

### Memory inventory (Phase 1 migration target)

| Memory | Type | Source | recall | Notes |
|---|---|---|---|---|
| `user-messages` | log | Channel-adapter ingress side effect | relevant | Historical user statements matching the current topic |
| `penny-messages` | log | `send_message` side effect | recent | Latest few things Penny sent |
| `browse-results` | log | `browse` tool side effect | off | Processing queue for knowledge extractor |
| `likes` | collection | Split from `preference` (valence=positive) | relevant | Preferences matching the current topic |
| `dislikes` | collection | Split from `preference` (valence=negative) | relevant | |
| `notified-thoughts` | collection | Split from `thought` (notified_at IS NOT NULL) | relevant | Past thoughts relevant to current topic |
| `unnotified-thoughts` | collection | Split from `thought` (notified_at IS NULL) | off | Working queue, not surfaced |
| `knowledge` | collection | Migration from `knowledge` table | relevant | Facts relevant to current topic; keyed by URL |

`user-messages` and `penny-messages` are separate physical logs. There's no virtual merged-view memory — agents that need chronological conversation context rely on the existing `message_log` thread chain (for reply threading) or pull from each log separately and merge client-side if needed. Ambient recall renders them as separate sections in the system prompt; the chat model stitches the conversation together in its head.

System tables (`runtime_config`, `promptlog`, `schedule`, `agent_cursor`, `media`, migrations, etc.) stay as first-class system tables — memories are *user-meaningful data*, not infrastructure.

### Existing → memory mapping, per agent

| Agent | Reads | Writes | Moves |
|---|---|---|---|
| History (prefs) | `user-messages` (log_read_since) | `likes` / `dislikes` (collection_write) | — |
| Thinking | `likes` (collection_read_random), `dislikes` (collection_read_all), `search` / `browse`, `exists` across thought collections | `unnotified-thoughts` (collection_write) | — |
| Notifier | `unnotified-thoughts` (collection_read_random), `penny-messages` (log_read_recent, optional) | `send_message` (→ penny-messages log side effect) | `unnotified-thoughts` → `notified-thoughts` |
| Knowledge extractor | `browse-results` (log_read_since) | `knowledge` (collection_write) | — |
| Chat | Ad-hoc reads of any memory (plus ambient context assembled by hypervisor from `user-messages`, `penny-messages`, and other recall-enabled memories; thread context via existing `message_log` chain) | `penny-messages` via `send_message`, user-authored writes to collections/logs | user-driven moves/deletes |

### Data-layer consequences

Phase 1 only changes the data layer and how agents read/write it. Agent behaviors themselves don't change (scheduling, rate limiting, cooldowns external to data, etc. stay as they are). A few behaviors DO change because they were implemented as mutable fields on the old tables — those fields don't exist in the new model. Don't re-implement them by reflex during migration.

- **Preference mention counts / thresholds.** Was a mutable counter on each preference row. New: no counters. `likes` contains all extracted preferences. If prioritization matters, it's derivable from `user-messages` similarity queries.
- **`notified_at` re-eligibility / 24h cooldown.** Was a mutable timestamp on each thought. New: notified thoughts are in the `notified-thoughts` collection; re-surfacing would require moving them back, which we don't do. The thinking agent produces fresh thoughts continuously; the notifier has new candidates.
- **Per-preference cooldowns.** Was derived from `notified_at` checks. New: same function served by the `exists` check against `notified-thoughts` — if topic already shared, exists trips, notifier exits without sending.
- **`last_thought_at` ordering for seed selection.** Was a mutable field on preference. New: `collection_read_random` is good enough. Anti-clustering via embedding similarity is a future optimization, not a primitive.
- **Knowledge aggregate-on-revisit** (existing summary + new page content → merged). Was an upsert path. New: second write with same URL key returns "duplicate found, write failed." Model moves on. Content diverges from the page if the page changes materially; acceptable tradeoff (delete + rewrite is the escape hatch).
- **Watermark-based batch reads in domain stores.** Each existing store had its own "latest processed timestamp" pattern. Replaced by the uniform `log_read_since` cursor mechanism.
- **Multi-strategy similarity dedup config (TCR_OR_EMBEDDING etc.).** Was a configurable knob in current code. Internal implementation detail of `collection_write` now; doesn't leak into the tool interface.

### Migration approach

1. Add the `memory` and `memory_entry` tables (plus `agent_cursor` and `media`).
2. Implement the access layer (`db.memories.*`, `db.media.*`) and the full tool surface.
3. Migrate existing data scripted, one-way — no dual-write. Each old table becomes its corresponding new memory per the inventory.
4. Rewire each agent to read/write via tools, one agent at a time (order: history → thinking → notifier → knowledge → chat). Old Python write paths get deleted as each agent migrates.
5. Rewire all ambient-injection call sites (chat context, dislikes injection, related messages retrieval, related knowledge retrieval) to use the unified per-memory recall mechanism.
6. Drop old specialized tables in a final migration.
7. Add user-facing chat-agent tools (`collection_create`, `log_create`, writes, updates, archive) to unlock the "life OS" capability.
8. Add the media store and swap the channel adapters to the token pattern (ingress: bytes → id → token in text; egress: token → bytes → channel attachment).
9. Update the browser addon UI: the current per-type views (Likes, Dislikes, Thoughts, etc.) become a single dynamic memory browser that lists every collection and log that exists and lets the user browse entries of any of them. Nothing is hardcoded per type.

Each step is independently deployable. Integration tests serve as the parity harness — existing tests should pass unmodified after each agent's migration (behavior preservation), with new tests added for anything that changed shape.

### Validation status

Dry-run harness at `data/collection-harness/` validates the tool surface and prompt shapes against gpt-oss:20b (Ollama, temperature 0). Four scenarios, all running end-to-end with stubbed tool implementations:

- **`preference_extraction.py`** (history agent) — `read_next` + batched per-collection `write` to likes/dislikes. Cycle: 4 turns. Correctly filters non-preferences, handles dedup rejections without retry, exits cleanly.
- **`thinking.py`** (thinking agent) — `read_random` from likes, `read_all` from dislikes, `search`, `browse`, `exists` cross-collection check, `write`. Cycle: 6–7 turns. Validates dislike-filtering in model-space; model even steers its browse choice away from dislike-adjacent content.
- **`notify.py`** (notifier) — `read_random` from unnotified-thoughts, `send_message`, `collection_move` to notified-thoughts. Cycle: 4 turns. Move primitive preserves key/content; empty-queue path exits in 2 turns.
- **`knowledge.py`** (knowledge extractor) — `read_next` from browse-results (cap=1 per invocation), single-entry `write` to knowledge. Cycle: 3 turns. Each invocation processes one page; pagination happens across runs via cursor.

All harnesses use narrow single-purpose tools, explicit `done()` exit, no stray text. Validates that:
- gpt-oss:20b handles the tool shapes reliably
- The batch-write shape (`entries=[...]`) is unambiguous
- `done()` discipline prevents summary bloat
- Cursor management stays entirely in Python
- Cross-collection `exists` is cleanly model-callable
- Move-based state transitions work end-to-end
- The prompt pattern (agent job + collection list + tool list) produces consistent behavior

---

## Phase 2: Task Framework (Behavior Layer)

Out of scope for detailed design here — we'll spec this after Phase 1 ships.

High-level shape: each existing agent, currently run by code-owned orchestration, becomes a *task* — a row in a `task` table with a user-editable system prompt, a schedule, a success criterion, and access to the same tool surface. Users author new tasks through conversation. The hardcoded agents in `penny/agents/*.py` are replaced by a task runner that resolves (prompt, tools) and runs the agent loop.

Because Phase 1 already moved every agent's data interactions into model-space tool calls, the Phase 2 migration is a mechanical lift: swap the Python orchestration shell for the task runner, move the system prompt into the `task` row, add success-criteria grading. No data-layer changes.

Key Phase 2 questions to defer:
- Success criteria format + grader prompt
- Task prompt versioning / edit history
- Event-triggered vs schedule-triggered tasks
- Self-correction loop (task edits its own prompt on repeated failure)
- Whether the chat agent eventually becomes a task too

---

## Downstream opportunities

Things that become easy or trivial once the new data layer (and eventually the task framework) land. Not scoped for the initial migration, but worth capturing so we remember what the platform unlocks.

### Unlocked by Phase 1 (data layer)

- **Profile collection + profile-extractor.** A new `profile` collection (recall: relevant) and a small agent (same shape as the preference extractor) that reads `user-messages` for facts about the user's life — relationships, activities, habits, routines, things they care about beyond like/dislike. Accumulates over time; surfaces via ambient recall when topically relevant. Gives Penny much richer personalization than preferences alone.
- **Reports collection.** A `reports` collection (recall: relevant; key = topic/title, content = long-form markdown). Penny writes reports over time — research summaries, pattern observations, analyses — and extends them as new material comes in. Pairs naturally with the long-form research-mode idea (user seeds a topic, Penny grows the report across many cycles, report becomes its own contextual anchor).
- **Rules / standing instructions collection.** User says "Penny, remember: never do X." Penny creates (or writes into) a `rules` collection with `recall: all`. Every entry is a rule or guideline that should always apply. Because recall is `all`, every rule gets injected on every turn unconditionally — no similarity match needed, they're persistent invariants. Small by design (tens of entries at most); the user prunes old rules as they become obsolete. Gives the user a natural way to shape Penny's behavior without editing a core prompt.

### Unlocked by Phase 2 (task framework)

- **Task run log + monitor task = fully internal self-correction.** Every task invocation writes to a `task-runs` log (success/fail + short run summary). A user-authored `monitor` task reads that log, flags tasks that fail repeatedly, and surfaces them for the user to correct via conversation — no git, no PR, no deploy. This is the same "tool outputs → log → user-composed loop" pattern applied to Penny's own execution: task runs are just another tool output, the monitor is just another task over a log, the corrections are just edits to task prompts. The self-improvement loop isn't special — it's the general pattern turned inward. penny-team implements the same shape today via GitHub issues and PRs; this is the fully-internal version.
- **Web watchers — user-authored via conversation.** User says "watch this page and let me know when X shows up." Penny creates a scheduled task: prompt = "browse this URL, check for X, if found and not previously notified, `send_message`." A small companion collection tracks what's already been flagged so nothing is reported twice. Works for stock/availability alerts, new listings on a marketplace, price drops, new posts on a watched blog, status changes on a service page. No new primitives needed — just browse + collection write + send_message wired together in a task prompt the user authored by talking.

---

## Guardrails

- **Every data op is a tool call.** No Python read/write logic outside `db.memories.*` and the tool implementations. Every agent's data footprint is visible in its tool-call log.
- **Content is immutable.** System-maintained fields (author, `created_at`, embeddings) are never changed after write. All semantic mutability is via `move` or delete+rewrite.
- **Cursor commits only on successful run completion.** At-least-once processing; dedup catches redundant reprocessing.
- **Subagent contexts stay narrow.** Ambient injection is for the outer chat agent only; background agents see only their declared system prompt and explicit tool-call returns.
- **Ambient injection volume is bounded.** Global K per memory with `recall` on; total context scales with the number of recall-enabled memories, not row counts.
- **`send_message` is always rate-limited.** Hypervisor enforces; not agent-visible.
- **System tables stay separate from memories.** `runtime_config`, `promptlog`, `schedule`, `agent_cursor` are infrastructure, not user data.

---

## Non-Goals

- Free-running meta-loop that picks tools from a large menu.
- Self-modifying outer chat agent or task framework in Phase 1.
- Multi-user isolation or grants — Penny is single-user.
- Filesystem-shaped data layer — collections are the affordance; no arbitrary paths or writable blobs.
- Enforced schemas or migrations on collection entries — schemas don't exist.
- Mutable entries — content is immutable; state is via collection membership.
- Task-to-task synchronous invocation — all inter-agent data flow is through collections.
- Re-implementing the old mutable-field patterns (mention counts, per-preference cooldowns, `notified_at` re-eligibility, etc.) during migration — their replacements are listed in *Data-layer consequences*.

---

## Open Questions

Most of the big design questions got settled during harness iteration. These are the remaining ones:

- **Agent identity propagation.** Python tool dispatchers need to know which agent is calling (for cursor scoping and author tagging). Probably a contextvar set by the orchestrator before the run. Settle during implementation.
- **Cursor table naming + schema.** `agent_cursor (agent_name, collection_name, last_read_at)` is the straw-man. Validate the key shape during Phase 1 design.
- **Global recall K.** Start with a system-wide default (probably 5). If any collection visibly over- or under-represents in chat context, promote to a per-collection override later — but not as a user-authoring concern; as infrastructure tuning.
- **Phase 2 triggering model.** Mostly out of scope, but: Phase 1 agent runs today are invoked by `BackgroundScheduler`. In Phase 2, this becomes a task runner. Decide at Phase 2 time whether tasks can be event-triggered (e.g., "new user-message entry" triggering history) or stay schedule-only.

---

## The Pitch

*"A personal AI you can reshape through conversation, built on primitives that can't betray you."*

The most exciting thing about the current wave of autonomous agents is that they reshape themselves — new behaviors, new data structures, new workflows, authored by talking to them. The most dangerous thing is that they typically do this through unrestricted shell, filesystem, and package-installation access. The capability and the vulnerability are the same mechanism.

Phase 1 separates the two at the data layer. The capability (user-authored storage, agent-broker'd data interactions, cooperative maintenance of the user's digital life) is preserved. The vulnerability (unrestricted system access, mutable-state sprawl, hidden persistence) is replaced with typed primitives: memories instead of filesystem, narrow tools instead of shell, immutable content instead of arbitrary mutation. Same flexibility, much smaller attack surface.

Phase 2 extends the separation to the behavior layer.

The current hardcoded pipelines become the Phase 1 benchmark: the MVP is done when every agent reproduces its behavior through nothing but collection tool calls. The harness already validates that it works with a small local model. After Phase 1 lands, every future memory is a `create_collection` or `create_log` call. After Phase 2, every future behavior is a `create_task` call. No PRs required.
