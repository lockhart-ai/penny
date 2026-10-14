# Memory Implementation Plan

Companion to `task-framework-plan.md` (design doc). This is the tactical migration plan: schema, access layer, tool registry, per-agent port, data migration.

Ordered in independently-deployable stages. Each stage ships behind the prior one; existing integration tests serve as the parity harness at every step.

---

## Stage 1: Data layer foundation

### Schema design — unified table, for now

One `memory` table for metadata (both collections and logs), one `memory_entry` table for entries (key nullable for logs).

We evaluated splitting into per-type tables (`collection_entry`, `log_entry`) for schema-encoded invariants. The trade-off: separate tables encode "collections always have keys, logs never do" at the DB level, but lose the uniform FK target that cursor/recall/archive infrastructure all want. Starting unified for operational simplicity; we can split later if the invariant violations become a real problem or we add a type that really diverges in shape.

```sql
CREATE TABLE memory (
  name          TEXT PRIMARY KEY,
  type          TEXT NOT NULL CHECK (type IN ('collection', 'log')),
  description   TEXT NOT NULL,
  recall        TEXT NOT NULL CHECK (recall IN ('off', 'recent', 'relevant', 'all')),
  archived      INTEGER NOT NULL DEFAULT 0,
  created_at    DATETIME NOT NULL
);

CREATE TABLE memory_entry (
  id                INTEGER PRIMARY KEY,
  memory_name       TEXT NOT NULL REFERENCES memory(name),
  key               TEXT,              -- null for log entries
  content           TEXT NOT NULL,
  author            TEXT NOT NULL,
  key_embedding     BLOB,              -- null if key is null
  content_embedding BLOB NOT NULL,
  created_at        DATETIME NOT NULL
);

CREATE INDEX ix_memory_entry_by_created ON memory_entry(memory_name, created_at);
CREATE INDEX ix_memory_entry_by_key     ON memory_entry(memory_name, key);

CREATE TABLE agent_cursor (
  agent_name   TEXT NOT NULL,
  memory_name  TEXT NOT NULL,
  last_read_at DATETIME NOT NULL,
  updated_at   DATETIME NOT NULL,
  PRIMARY KEY (agent_name, memory_name)
);

CREATE TABLE media (
  id          INTEGER PRIMARY KEY,
  mime_type   TEXT NOT NULL,
  bytes       BLOB NOT NULL,
  source_url  TEXT,
  created_at  DATETIME NOT NULL
);
```

### Access layer

`penny/database/memory_store.py` — single module exposing `MemoryStore` on the database facade (`db.memories`).

```python
class MemoryStore:
    # Metadata
    def create_collection(self, name, description, recall, archived=False) -> Memory
    def create_log(self, name, description, recall, archived=False) -> Memory
    def get(self, name) -> Memory | None
    def list(self) -> list[Memory]
    def archive(self, name) -> None
    def unarchive(self, name) -> None

    # Collection writes
    def write(self, name, entries: list[EntryInput], author: str) -> list[WriteResult]
    def update(self, name, key, content, author) -> Literal["ok", "not_found"]
    def move(self, key, from_name, to_name, author) -> Literal["ok", "not_found", "collision"]
    def delete(self, name, key) -> int  # rows removed

    # Log writes
    def append(self, name, entries: list[LogEntryInput], author: str) -> None

    # Reads
    def get_entry(self, name, key) -> list[Entry]                  # collection only, list because unique-key still needs protection
    def read_latest(self, name, k) -> list[Entry]
    def read_recent(self, name, window_seconds, cap=None) -> list[Entry]
    def read_since(self, name, cursor: datetime, cap=None) -> list[Entry]
    def read_random(self, name, k) -> list[Entry]
    def read_similar(self, name, anchor, k) -> list[Entry]
    def read_all(self, name) -> list[Entry]
    def keys(self, name) -> list[str]

    # Introspection
    def exists(self, names: list[str], key: str | None, content: str, thresholds: DedupThresholds) -> bool
```

Internal helpers:
- `_embed(text)` — wraps the embedding client
- `_check_dedup(memory_name, key, content)` — similarity check on both axes
- Type-enforcement: `write/update/move/delete` assert memory type is 'collection'; `append` asserts 'log'. Raises `MemoryTypeError` on mismatch.

### Dedup rule

Collections dedup on write. Three signals are evaluated against each existing entry:

1. `token_containment_ratio(candidate.key, existing.key)` — lexical; cheap; catches "dark roast" vs "dark roast coffee" and year-stripped variants the embedding drifts on
2. `cosine(candidate.key_embedding, existing.key_embedding)` — paraphrase
3. `cosine(candidate.content_embedding, existing.content_embedding)`

A candidate is a duplicate if **any one signal meets its strict threshold**, OR if **any two signals meet their relaxed thresholds**. Strict catches obvious on-one-axis matches (same key, identical content); relaxed catches weak-everywhere matches a single-signal gate misses.

Signals that can't be computed (missing keys on log entries, missing embeddings when no model is configured) are skipped — the rule degrades to "whatever's comparable fires."

Starting thresholds (all live in `PennyConstants`):
- `MEMORY_KEY_TCR_STRICT_THRESHOLD = 1.0` / `MEMORY_KEY_TCR_RELAXED_THRESHOLD = 0.65` (abbreviation band — 2/3 is a common score and 0.67 literal misses it by a bit)
- `MEMORY_KEY_SIM_STRICT_THRESHOLD = 0.90` / `MEMORY_KEY_SIM_RELAXED_THRESHOLD = 0.75`
- `MEMORY_CONTENT_SIM_STRICT_THRESHOLD = 0.90` / `MEMORY_CONTENT_SIM_RELAXED_THRESHOLD = 0.75`

Any hit rejects the write with `"duplicate found, write failed"` — the model doesn't need to reason about which axis triggered. Tune in production by watching false-positive / false-negative rates on real writes. This is a consolidation of the several bespoke dedup strategies in the old architecture; per-memory tuning happens via override `DedupThresholds` passed into `write()`.

### Cursor helpers

`penny/database/cursor_store.py`:

```python
class CursorStore:
    def get(self, agent_name, memory_name) -> datetime | None
    def advance_committed(self, agent_name, memory_name, timestamp) -> None
    # No pending/rollback at the DB layer — that's orchestration-layer concern
```

### Media store

`penny/database/media_store.py`:

```python
class MediaStore:
    def put(self, bytes_, mime_type, source_url=None) -> int  # returns id
    def get(self, id) -> MediaEntry | None
```

### Migration

New migration file `0025_memory_schema.py` creates the tables above. No data migration yet — just schema.

---

## Stage 2: Tool registry

`penny/tools/memory_tools.py` — tool implementations as `Tool` subclasses matching the existing tool framework.

### Tool set

Each tool is a thin wrapper that:
1. Validates args via a Pydantic model (per CLAUDE.md rules)
2. Calls `db.memories.*`
3. Returns a result suitable for serialization back to the model

Collection tools:
- `CollectionCreate` → `db.memories.create_collection`
- `CollectionGet` → `db.memories.get_entry`
- `CollectionReadLatest` → `db.memories.read_latest` (cap from config)
- `CollectionReadRandom` → `db.memories.read_random(k)`
- `CollectionReadSimilar` → `db.memories.read_similar(anchor, k)`
- `CollectionReadAll` → `db.memories.read_all`
- `CollectionKeys` → `db.memories.keys`
- `CollectionWrite` → `db.memories.write`
- `CollectionUpdate` → `db.memories.update`
- `CollectionDelete` → `db.memories.delete`
- `CollectionMove` → `db.memories.move`
- `CollectionArchive` → `db.memories.archive`
- `CollectionUnarchive` → `db.memories.unarchive`

Log tools:
- `LogCreate` → `db.memories.create_log`
- `LogReadLatest`, `LogReadRecent`, `LogReadSimilar`, `LogReadAll` — wrappers
- `LogReadNext` — reads since the agent's cursor, Python-managed cap, records pending advance
- `LogAppend` — most agents won't need this; used by side-effect wrappers and user-collection writes

Discovery / introspection:
- `ListMemories` → `db.memories.list`
- `Exists` → `db.memories.exists(names, key, content)`

Lifecycle:
- `Done` — empty body, signals the orchestration loop to exit (no model-facing content change)

### Author attribution

The tool dispatcher reads `current_agent()` from a contextvar set by the orchestration layer before each run. All writes/moves/deletes/appends stamp `author` with that value. The model never sees or sets author.

### Cursor pending/commit

`LogReadNext` has a two-phase commit handled by the run orchestrator:
1. Each call records the batch's max `created_at` in an in-memory pending dict keyed by `(agent_name, memory_name)`.
2. On successful run completion, the orchestrator calls `CursorStore.advance_committed` for each pending entry.
3. On run failure, pending is discarded.

### Failure recovery in the agent loop

The orchestration layer (`penny/agents/base.py` or equivalent) handles three recurring model-fumble failure modes. All three were validated against gpt-oss:20b in the dry-run harness.

1. **Malformed JSON in tool call arguments** — when `client.chat.completions.create` raises `openai.InternalServerError` with text containing "error parsing tool call" or "invalid character" (Ollama's parser choked on the model's output), append a corrective user message (`"Your previous tool call had malformed JSON. Retry with strictly-valid JSON in the arguments."`) and retry the same step.

2. **Empty response with no tool calls** — when the model returns a response with neither `tool_calls` nor meaningful `content`, it's dropped the thread. Append a nudge (`"You returned an empty response with no tool calls. Continue and complete the user's request — if multiple changes are needed, make all of them before responding."`) and retry.

3. **Bad tool call arguments** — wrap `dispatch_tool` in try/except; catch `TypeError`, `pydantic.ValidationError`, and `MemoryTypeError`; return the error message as a string tool result (`"error: bad arguments for collection_update: missing 'content'"`). The model sees the error in the next turn and can retry with corrected args.

Each retry counts against `max_steps`; if the agent exhausts its budget on retries, that's a legitimate failure and the run is logged as failed.

---

## Stage 3: Ambient recall assembly

`penny/agents/recall.py`:

```python
async def build_recall_block(
    db, current_message: str, k_default: int = 5, similarity_floor: float = 0.35
) -> str:
    """For each non-archived memory with recall != 'off', render the appropriate slice."""
```

Logic per memory:
- `recall: off` — skip
- `recall: recent` — `read_latest(k_default)`
- `recall: relevant` — compute embedding of `current_message`, `read_similar(current_message, k_default)` filtered by similarity >= floor
- `recall: all` — `read_all`

### Conversation-pair rendering

The assembler has one special case: when both `user-messages` and `penny-messages` appear in the recall set for a given turn, it merges their entries chronologically into a single "Conversation" section rather than rendering them as two separate sections. The model reads one interleaved transcript instead of stitching two disjoint lists together in its head.

This is a display-layer concern only — the data layer still has two independent physical logs. No virtual memory, no merged storage. `build_recall_block` detects the pair in its rendering pass and interleaves by `created_at`. Each log still contributes its own K-cap independently (so you get up to K user messages + K Penny messages merged).

Example output when both logs are recall-enabled:

```
## Recall (entries relevant to this message)

Conversation (most recent turns, both sides):
  [2026-04-20 09:12] user: hey have you seen this cyberpunk novel
  [2026-04-20 09:13] penny: yeah, the one with the ecological worldbuilding?
  [2026-04-20 09:15] user: yeah exactly
  ...

Log: user-messages — historically relevant user statements
  - (2 weeks ago) "loved the worldbuilding in that last novel"
  - ...

Collection: london-trip (Trip to London)
  - [tuesday] WeWork Shoreditch High Street...
```

The "historically relevant user statements" section comes from the `relevant` mode on `user-messages` (past user statements matching the current topic by similarity). The "Conversation" section comes from the `recent` mode on both logs, merged by the pair rule. Both can coexist because `user-messages` gets evaluated under both its `relevant` mode AND the conversation-pair rule when it applies.

Actually, to avoid the complexity of the same log appearing in two recall passes, the simpler rule: `user-messages` is `recall: relevant` (historical matches) and `penny-messages` is `recall: recent` (latest sends). The conversation-pair rule kicks in only when `user-messages` is ALSO rendered in recent mode as part of the pair — which happens when the assembler detects the pair and pulls recent from both. Implementation detail: the assembler can treat the pair as a special unit that pulls recent from both AND also lets `user-messages` do its normal relevant pass. The result is the user sees both a "conversation" section (recent merged) and a "historically relevant user statements" section (relevant).

### Memory registration

The pair is configured in code, not in the memory table. A simple constant in `recall.py`:

```python
CONVERSATION_PAIRS = [("user-messages", "penny-messages")]
```

### Scope

Chat agent's `_build_system_prompt` appends this block after its existing structure. Background agents do NOT call `build_recall_block` — they see only their declared tool results.

---

## Stage 4: History agent port

`penny/agents/history.py`

### Current pattern

- Scheduler invokes `HistoryAgent.execute()`
- Reads unprocessed prompts via `db.knowledge.get_latest_prompt_timestamp` + `db.messages.get_prompts_with_browse_after`
- Calls LLM with `response_format=PreferenceList`, parses structured output
- Writes to `db.preferences.add` for each extracted preference
- Similar for knowledge extraction (see Stage 7)

### New pattern — preference extraction

Python orchestration shell (tiny):

```python
async def run_preference_extractor(self):
    await self.llm.run_agent(
        system_prompt=PREFERENCE_SYSTEM_PROMPT,
        tools=[LogReadNext, CollectionWrite, Done],
        agent_name="preference-extractor",
    )
```

That's the whole thing. No JSON parsing, no structured-output decoding, no Python-side dedup logic. All of that moves into:

- `LogReadNext("user-messages")` — Python-managed cursor
- `CollectionWrite("likes"|"dislikes", entries=[...])` — dedup baked in

The agent's system prompt (from the validated harness):

```
You extract the user's likes and dislikes from their recent messages.

1. Call log_read_next("user-messages") to fetch messages you haven't seen yet.
2. Identify every genuine preference across the returned messages.
3. Call collection_write once per target collection (likes, dislikes) batching all entries.
4. Call done().
```

Entry shape — `key`: the topic (short phrase), `content`: the user's raw message verbatim.

### Removed

- `PreferenceList` Pydantic schema
- `_dedup_preference_topics`, `_bump_existing_mentions` (dedup now in tool layer)
- Any `mention_count` handling

---

## Stage 5: Thinking agent port

`penny/agents/thinking.py`

### New pattern

```python
async def run_thinking_cycle(self):
    await self.llm.run_agent(
        system_prompt=THINKING_SYSTEM_PROMPT,
        tools=[
            CollectionReadRandom, CollectionReadAll,
            Search, Browse,
            Exists,
            CollectionWrite,
            Done,
        ],
        agent_name="thinking-agent",
    )
```

System prompt (from the validated harness, generic-ized):

```
You are Penny's thinking agent. Once per run:
1. collection_read_random("likes", 1) — pick one seed topic.
2. collection_read_all("dislikes") — see what the user doesn't like.
3. search + browse — find something timely and interesting grounded in real results.
4. Draft ONE thought connecting what you found to the user's interest.
5. Check it against dislikes. If conflict, call done() without writing.
6. exists(["unnotified-thoughts", "notified-thoughts"], key, content). If true, call done().
7. collection_write("unnotified-thoughts", entries=[{key, content}]).
8. done().
```

### Removed

- `_pick_seeded_prompt` (seed selection via model)
- `_find_duplicate_thought` (dedup via tool layer + exists)
- `_matches_dislike` (dislike filter done in model)
- `last_thought_at` tracking
- Multi-strategy similarity knobs
- All JSON schema parsing for thought output

---

## Stage 6: Notifier port

`penny/agents/notify.py`

### New pattern

```python
async def run_notifier(self):
    await self.llm.run_agent(
        system_prompt=NOTIFIER_SYSTEM_PROMPT,
        tools=[
            CollectionReadRandom,
            SendMessage,        # existing, unchanged interface; internal side-effect now appends to penny-messages log
            CollectionMove,
            Done,
        ],
        agent_name="notifier",
    )
```

System prompt (from the validated harness):

```
You are Penny's notifier. Once per run:
1. collection_read_random("unnotified-thoughts", 1) to pick a candidate.
2. If empty, call done().
3. Otherwise, send_message(content=<the thought's content>) — send verbatim.
4. collection_move(key=<the thought's key>, from_collection="unnotified-thoughts", to_collection="notified-thoughts").
5. done().
```

### Removed / changed

- Per-preference cooldown logic (replaced by move-to-notified + exists-check in thinking)
- `notified_at` timestamp handling (state is collection membership now)
- `_get_top_thoughts` scoring (single random pick is enough)
- Exponential backoff lives in `send_message` tool implementation, unchanged

---

## Stage 7: Knowledge extractor port

`penny/agents/history.py` (knowledge extraction currently lives here; split into its own file or keep in history — either works).

### New pattern

```python
async def run_knowledge_extractor(self):
    await self.llm.run_agent(
        system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
        tools=[LogReadNext, CollectionWrite, Done],
        agent_name="knowledge-extractor",
        read_cap_override={"browse-results": 1},   # one page per invocation
    )
```

System prompt:

```
You are Penny's knowledge extractor. You process ONE page per run.

1. log_read_next("browse-results"). Returns at most one unprocessed page; content begins with "URL: <url>".
2. If empty, done().
3. Draft ONE knowledge entry:
   - key: a short, specific title
   - content: 3–8 sentence self-contained summary. Include "Source: <url>" at the end.
4. collection_write("knowledge", entries=[{key, content}]).
5. done().
```

### Removed

- `_dedup_browse_results_by_url` (log preserves all fetches; dedup happens at knowledge write based on title key)
- `_normalize_url` (URL is in content body; normalization for key is a caller concern — can be added back later if needed, but not critical for correctness)
- `_aggregate_knowledge` (no upsert; second fetch of same URL either produces a different title and writes anew, or same title and dedup rejects)

---

## Stage 8: Chat agent port

`penny/agents/chat.py` — the heaviest port because chat touches many memories and relies on ambient recall.

### New pattern

```python
async def handle(self, message):
    # ambient recall assembled in system prompt — see Stage 3
    await self.llm.run_agent(
        system_prompt=self._build_system_prompt(message),
        tools=[
            # full tool surface — chat is the generalist
            ListMemories,
            CollectionCreate, CollectionGet, CollectionReadLatest, CollectionReadRandom,
            CollectionReadSimilar, CollectionReadAll, CollectionKeys,
            CollectionWrite, CollectionUpdate, CollectionDelete, CollectionMove,
            CollectionArchive, CollectionUnarchive,
            LogCreate, LogReadLatest, LogReadRecent, LogReadSimilar, LogReadAll,
            Search, Browse,
            SendMessage,
            # Note: no Done — chat agent terminates when it emits a response with no tool_calls,
            # and that final message IS the reply to the user
        ],
        agent_name="chat",
    )
```

### System prompt structure (extending existing template-method pattern)

Existing building blocks (`_identity_section`, `_profile_section`, etc.) stay. New additions:
- `_recall_section` — calls `build_recall_block`
- `_memories_section` — injects the memory registry (name + description + type + recall for each non-archived memory). This is what helps Penny know what collections exist when users reference them.

Drop:
- `_related_knowledge_section` (replaced by ambient recall on the knowledge collection)
- `_related_messages_section` (replaced by ambient recall on the messages log)
- `_dislike_section` (replaced by ambient recall on the dislikes collection)

### Removed

- `db.knowledge.get_related` direct calls in context assembly
- `db.preferences.get_negative_preferences` direct calls
- `db.messages.get_related_messages` direct calls
- All hand-coded "similarity sort and inject" logic

---

## Stage 9: Side-effect wiring

Three places the system writes memories automatically, not via model tool calls:

### Channel ingress → `user-messages` log

`penny/channels/base.py:handle_message` — after existing `log_message` call, also `db.memories.append("user-messages", [{content: msg.content}], author="user")`.

### `send_message` → `penny-messages` log

The tool's implementation, after delivering to the channel, also `db.memories.append("penny-messages", [{content}], author=current_agent())`.

### `browse` → `browse-results` log

The browse tool's implementation, after fetching the page, also `db.memories.append("browse-results", [{content: f"URL: {url}\n\n{page_text}"}], author=current_agent())`.

### Logging new tools is the default going forward

When someone adds a new tool to Penny in the future, the default should be: the tool side-effect-writes its output to a log (`<tool-name>-results` or similar). This is what makes user-authored loops possible — the user can say "analyze all my X" as long as `X` is already a log. Only skip the log for truly ephemeral tools with no plausible future consumer. This is the extension of the principle the agent core implements for `browse`, `send_message`, and channel ingress.

---

## Stage 10: Data migration

One-shot scripts, run during the migration cutover deploy. Each migrates one table into the new memory schema.

`penny/database/migrations/0026_memory_data_migration.py`:

1. Create memories (INSERT into `memory` table):
   - `user-messages` (log, recall=relevant)
   - `penny-messages` (log, recall=recent)
   - `browse-results` (log, recall=off)
   - `likes` (collection, recall=relevant)
   - `dislikes` (collection, recall=relevant)
   - `unnotified-thoughts` (collection, recall=off)
   - `notified-thoughts` (collection, recall=relevant)
   - `knowledge` (collection, recall=relevant)

2. Populate `memory_entry` from old tables:
   - `message_log` WHERE direction='inbound' → `user-messages`
   - `message_log` WHERE direction='outbound' → `penny-messages`
   - `preference` WHERE valence='positive' → `likes` (key = content if short enough; else a truncated slug)
   - `preference` WHERE valence='negative' → `dislikes`
   - `thought` WHERE notified_at IS NULL → `unnotified-thoughts` (key = title)
   - `thought` WHERE notified_at IS NOT NULL → `notified-thoughts`
   - `knowledge` → `knowledge` (key = title; URL goes into content body)
   - `browse-results` — no historical migration; starts fresh post-deploy

3. Existing embeddings on preferences, thoughts, knowledge carry over as `content_embedding`. Where a key is present, compute `key_embedding` in a batched pass.

4. The old tables remain in place (see Stage 12).

### Dual-write semantics post-migration

Important distinction in what happens to old tables after their corresponding agent is ported:

- **`message_log`** — keeps receiving writes indefinitely. The channel adapter writes incoming messages here (for device/routing state that lives on these rows) AND side-effect-writes to `user-messages` / `penny-messages` (for the memory layer). Dual-write, long-lived.
- **`preference`, `thought`, `knowledge`** — frozen at migration time. Once the agent that wrote to each is ported (Stages 4-8), that agent stops writing to the old table. The old tables keep the snapshot from migration-time and receive no further writes.

This asymmetry matters for rollback: reverting a message-related change is safe indefinitely since `message_log` is always current; reverting a preference/thought/knowledge-related change loses any writes that happened to the new memory after the agent was ported. Verify at each port-stage PR that the old-to-new migration captured everything relevant before the agent rewires.

---

## Stage 11: Agent rewire

Agents ported one at a time in a PR per agent. Order:
1. History (preferences)
2. Thinking
3. Notifier
4. Knowledge extractor
5. Chat

After each agent is ported, its existing integration tests must pass unmodified. Test failures indicate parity issues to fix before merging.

---

## Stage 12: Old tables — keep in place for now

We are not dropping `preference`, `thought`, `knowledge`, or `message_log` as part of this migration. Once all agents are ported and the new memories are the sole read/write path from agents, the old tables are dead weight — but dead weight is cheap, and keeping them gives us:

- A fallback reference during the port if anything's wrong with migrated data
- Ability to run an old-vs-new comparison at any point post-migration
- Zero risk from an overeager drop that removes data we later need

Drop is a later, independent decision. It can happen weeks/months after the port when we're confident. At that point it's a one-line migration.

`message_log` specifically will likely stay long-term regardless, since the channel layer uses it for device/routing and there's no urgency to move that into the memories.

---

## Stage 13: Add user-facing collection tools to chat agent

At this point the chat agent has the tools but they may have only been exercised with existing collections. Now:
- User says "track my weight" → chat agent calls `log_create` or `collection_create` based on intent
- User says "remember this rule" → chat agent calls `collection_create` with `recall: all` + writes the rule
- etc.

No new tools — just documented behavior validated with real user interactions.

---

## Stage 14: Browser addon UI update

Addon currently has hardcoded per-type tabs (Likes, Dislikes, Thoughts). Replace with:
- Dynamic list of all non-archived memories (from `list_memories()` via the WebSocket protocol)
- Click a memory → browse its entries (paginated for logs, full for collections typically)
- Filter/search by key or content
- No hardcoded per-type handling

Addon protocol gains a `list_memories` and `read_memory_entries(memory, shape)` request.

---

## Validation per stage

**Stages 1–3 (data layer, tools, recall)**: unit tests on the new access layer + tool argument validation. No agent behavior changes yet.

**Stages 4–8 (per-agent port)**: existing integration tests in `penny/tests/agents/` must pass unmodified. If they don't, the port isn't complete.

**Stage 10 (data migration)**: test harness runs the migration against a copy of the production DB (via `make migrate-test`), verifies row counts match and spot-checks content integrity.

**Stage 11 (agent rewire)**: full integration test pass after the last agent is ported — confirms the system runs end-to-end with the new memories as the sole agent-facing data layer.

---

## Rollback

Each stage is behind its own migration number. Since we keep old tables throughout the migration (see Stage 12), rollback at any point is cheap — old tables are still populated via the pre-migration writes from each agent, so reverting an agent PR falls back to the original behavior without data loss.

Once we eventually decide to drop the old tables (separate, post-migration PR), that's the one irreversible step — which is exactly why it's separate and deferred.

---

## Ship order

1. Stage 1 (schema) — PR, deploy
2. Stage 2 (tools) — PR, deploy
3. Stage 3 (recall assembly) — PR, deploy (unused until chat port)
4. Stage 9 (side-effect wiring) — PR, deploy (writes to new memories; old tables still being written to as well)
5. Stage 10 (data migration) — PR, deploy; new memories populated from existing tables
6. Stages 4-8 (agent ports, one PR each, in the order listed)
7. Stages 13-14 (user collections, addon UI) — PRs, deploy as ready

Each stage is independently revertable. Old tables stay in place (see Stage 12) — drop is a later, separate decision when we're confident the new memories are the sole source of truth and we don't want the fallback anymore.
